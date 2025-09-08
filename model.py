
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import List, Tuple, Optional, Dict, Any

class StackedLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1,
        num_classes: int = 29,
    ):
        super().__init__()
        assert num_layers >= 2, "本实现默认两层以上（你要的是两层）"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,
            bidirectional=bidirectional,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * self.num_directions,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,
            bidirectional=bidirectional,
        )

        self.inter_drop = nn.Dropout(dropout)  # 仅用于定长分支；变长分支就不在中间丢弃
        self.head_drop = nn.Dropout(dropout)

        fc_in = hidden_dim * self.num_directions
        self.fc = nn.Linear(fc_in, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0.0)
            elif "fc" in name and "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "fc" in name and "bias" in name:
                nn.init.constant_(param.data, 0.0)

    def _flatten_hn(self, h_n: torch.Tensor) -> torch.Tensor:
        # h_n: (D, B, H) -> (B, D*H)
        return h_n.transpose(0, 1).contiguous().view(h_n.shape[1], -1)

    def forward(
        self,
        x: torch.Tensor,                 # (B, T, C)
        lengths: Optional[torch.Tensor] = None  # (B,)
    ) -> torch.Tensor:
        B, T, C = x.shape
        x = x.permute(1, 0, 2).contiguous()  # (T, B, C)

        if lengths is not None:
            # 变长：不手动排序，直接让 PyTorch 处理
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), enforce_sorted=False
            )
            _, (h1, _) = self.lstm1(packed)
            # 直接把上一层的 PackedSequence 结果继续喂给第二层（注意：传 h_ 不行，要传输出）
            # 正确做法：需要用 lstm 的输出，而不是 h1；所以我们得真正拿到 out1
            out1, _ = self.lstm1(packed)
            out2, (h2, _) = self.lstm2(out1)

            # 用第二层的 h_n 作为“多对一”表示
            feats = self._flatten_hn(h2)          # (B, D*H)
            feats = self.head_drop(feats)
        else:
            # 定长：常规张量流
            out1, _ = self.lstm1(x)               # (T, B, H*D)
            out1 = self.inter_drop(out1)
            out2, (h2, _) = self.lstm2(out1)      # h2: (D, B, H)

            feats = self._flatten_hn(h2)          # (B, D*H)
            feats = self.head_drop(feats)

        logits = self.fc(feats)                   # (B, num_classes)
        return logits
# ========================================================================
# ==============================
# 1) 将单个 fold 的 dict 适配为 PyTorch Dataset
# ==============================

class IMUKFoldDataset(Dataset):
    """
    期望 split_dict 形如：
    {
      "X_imu": List[List[window]],   # 外层：样本；内层：该样本的所有窗口
                                     # window 可是 np.ndarray 或 (np.ndarray, meta)
                                     # window 的数据应是 (L, F)
      "y": List[str|int],            # 与“样本”同长度，每个样本的手势标签
      # 可选：
      # "valid_lengths": List[List[int]]  # 与 X_imu 同结构，如果有就用，否则从窗口形状推
    }
    使用方式：
      - 若所有窗口 L 相同：可直接用 DataLoader 默认 collate（会堆成 (B, L, F)）
      - 若窗口 L 不相同：请在 DataLoader 传入 collate_fn=IMUKFoldDataset.collate_pad
                          这样会返回 (B, Lmax, F)、lengths(B,)
    """
    def __init__(self, split_dict: Dict[str, Any], label_map_gesture: Dict[str, int]):
        super().__init__()
        X_nested = split_dict["X_imu"]     # List[List[window]]
        y_samples = split_dict["y"]        # List[...]，与“样本”同长度
        assert len(X_nested) == len(y_samples), "X_imu 外层长度必须与 y 一致"

        # ===== 展平：把所有“窗口”变成样本 =====
        feats: List[np.ndarray] = []
        labels: List[int] = []
        metas:  List[Any] = []
        lengths_nested = split_dict.get("valid_lengths", None)

        for i, (windows, y_i) in enumerate(zip(X_nested, y_samples)):
            y_idx = label_map_gesture[str(y_i)]
            # 该样本的所有窗口
            for j, w in enumerate(windows):
                if isinstance(w, (list, tuple)):
                    data = w[0]
                    meta = w[1] if len(w) > 1 else None
                else:
                    data = w
                    meta = None

                a = np.asarray(data, dtype=np.float32)
                if a.ndim != 2:
                    raise ValueError(f"样本{i}的第{j}个窗口不是二维矩阵，shape={a.shape}")
                # 统一为 (L, F)：如果给的是 (F, L) 且 F<L 很可能通道在 0 轴，转置
                if a.shape[0] < a.shape[1]:
                    # 假设 (F, L) -> (L, F)
                    a = a.T
                L, F = a.shape
                feats.append(a)
                labels.append(y_idx)
                metas.append(meta)

        self.meta = metas

        # ===== 判断窗口是否等长 =====
        lengths = [arr.shape[0] for arr in feats]
        self.feature_dim = feats[0].shape[1]
        if not all(arr.shape[1] == self.feature_dim for arr in feats):
            raise ValueError(f"所有窗口的特征维 F 必须一致；前几个形状={[x.shape for x in feats[:5]]}")

        self.equal_length = len(set(lengths)) == 1
        if self.equal_length:
            # 直接堆叠成 (N, L, F)，DataLoader 默认即可
            X = np.stack(feats, axis=0).astype(np.float32, copy=False)
            self.X = torch.from_numpy(X)                           # (N, L, F)
            self.lengths = torch.full((self.X.size(0),), self.X.size(1), dtype=torch.long)
        else:
            # 保留为 list，配合自定义 collate_fn 做 pad
            self.X = [torch.from_numpy(x.copy()) for x in feats]   # List[(L_i, F)]
            self.lengths = torch.tensor(lengths, dtype=torch.long)

        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.equal_length:
            x = self.X[idx]                 # (L, F)
            y = self.y[idx]
            return x, y
        else:
            # 返回可变长样本，交由 collate_pad 处理
            x = self.X[idx]                 # Tensor (L_i, F)
            y = self.y[idx]
            l = self.lengths[idx]
            return {"x": x, "y": y, "length": l}

# ==============================
# 2) 标签映射工具：字符串 -> 索引
# ==============================
def build_label_map(values: list) -> dict[str, int]:
    """将标签（可能是字符串）映射为连续整数索引"""
    uniq = sorted({str(v) for v in values})
    return {k: i for i, k in enumerate(uniq)}

def _compute_macro_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    """
    基于混淆矩阵计算 macro recall 和 macro F1（多分类）
    y_true, y_pred: (N,) 的 LongTensor（已在 CPU 上）
    """
    with torch.no_grad():
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        tp = cm.diag()                                  # (K,)
        support = cm.sum(dim=1)                         # row sum = TP+FN
        pred_pos = cm.sum(dim=0)                        # column sum = TP+FP

        eps = 1e-12
        recall_per_class = (tp.float() / (support.float() + eps))           # (K,)
        precision_per_class = (tp.float() / (pred_pos.float() + eps))       # (K,)
        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

        macro_recall = recall_per_class.mean().item()
        macro_f1 = f1_per_class.mean().item()
        return macro_recall, macro_f1

# ==============================
# 3) K 折训练主流程（带早停）
# ==============================
def train_kfold(
    datasets: list[dict[str, dict]],
    num_epochs: int = 100,          # 默认训练 100 轮（早停会提前结束）
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    early_stopping: bool = True,    # 开启早停
    patience: int = 10,             # 容忍多少轮没有提升
    min_delta: float = 1e-4,        # 最小提升幅度
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results = []

    for fold_id, fold in enumerate(datasets, 1):
        print(f"\n========== Fold {fold_id} / {len(datasets)} ==========")
        train_split = fold["train"]
        val_split   = fold["val"]

        # —— 手势标签映射（基于训练集构建）——
        label_map_g = build_label_map(train_split["y"])
        num_gestures = len(label_map_g)

        # —— DataLoader ——
        train_ds = IMUKFoldDataset(train_split, label_map_g)
        val_ds   = IMUKFoldDataset(val_split,   label_map_g)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

        # —— 模型 ——
        input_dim = 6
        num_classes = 29
        model = StackedLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            bidirectional=False,  # 若想更强一些可置 True，同时 FC 输入会自动变为 2*hidden_dim
            dropout=0.2,
            num_classes=num_classes,
        ).to(device)

        # 优化器：Adam
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # 损失函数：交叉熵（对应你给的 H(p, q)；这里直接用 logits + 类别 id）
        criterion = nn.CrossEntropyLoss()

        # —— 早停相关变量 ——
        best_acc = 0.0
        best_val_loss = float("inf")
        best_recall = 0.0
        best_f1 = 0.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            # —— 训练 ——
            model.train()
            total_loss, n = 0.0, 0
            for x, y_g in train_loader:
                x, y_g= x.to(device), y_g.to(device)
                optimizer.zero_grad()
                logits_g = model(x)
                loss = criterion(logits_g, y_g)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = total_loss / max(1, n)

            # —— 验证 ——
            model.eval()
            val_loss, correct_g, tot = 0.0, 0, 0
            all_true, all_pred = [], []  # 新增：收集标签与预测
            with torch.no_grad():
                for x, y_g in val_loader:
                    x, y_g = x.to(device), y_g.to(device)
                    lg = model(x)
                    loss = criterion(lg, y_g)
                    val_loss += loss.item() * x.size(0)
                    pred_g = lg.argmax(1)
                    correct_g += (pred_g == y_g).sum().item()
                    tot += x.size(0)

                    # 收集到 CPU 上，便于后续构建混淆矩阵
                    all_true.append(y_g.detach().cpu())
                    all_pred.append(pred_g.detach().cpu())

            val_loss /= max(1, tot)
            acc_g = correct_g / max(1, tot)
            # 计算 macro recall & macro F1
            all_true = torch.cat(all_true, dim=0)
            all_pred = torch.cat(all_pred, dim=0)
            macro_recall, macro_f1 = _compute_macro_recall_f1(all_true, all_pred, num_classes=num_gestures)

            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
                  f"| val_loss={val_loss:.4f} | gesture_acc={acc_g:.3f}"
                  f"| macro_recall={macro_recall:.3f} | macro_f1={macro_f1:.3f}")


            # —— 检查是否刷新最佳 ——
            if acc_g > best_acc + min_delta:
                best_acc = acc_g
                best_val_loss = val_loss
                best_recall = macro_recall
                best_f1 = macro_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # —— 提前停止 ——
            if early_stopping and epochs_no_improve >= patience:
                print(f"早停触发！在第 {epoch} 轮停止，最佳 acc={best_acc:.3f}")
                break

        # —— 恢复并保存最佳模型 ——
        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(model.state_dict(), f"best_model_fold{fold_id}.pt")
            print(f"Fold {fold_id} 最优模型已保存 (val_acc={best_acc:.3f})")

        fold_results.append((best_acc, best_val_loss, best_recall, best_f1))

    return fold_results

# ==============================
# 4) 训练入口
# ==============================
def run_with_your_kfold(datasets):
    """
    直接把 build_kfold_datasets 的返回传进来即可：
        datasets, selected_categories = build_kfold_datasets(n_splits=5)
        run_with_your_kfold(datasets)
    """
    results = train_kfold(
        datasets=datasets,
        num_epochs=100,     # 设大一点，早停会提前终止
        batch_size=64,
        lr=2e-3,
        weight_decay=1e-4,
        early_stopping=True,
        patience=10,
        min_delta=1e-4
    )
    results = np.array(results)  # shape: (folds, 2) -> (acc, loss)
    accs = results[:, 0]
    losses = results[:, 1]
    recalls = results[:, 2]
    f1s     = results[:, 3]

    acc_mean = accs.mean() * 100
    acc_std  = accs.std(ddof=1) * 100
    loss_mean = losses.mean()
    loss_std  = losses.std(ddof=1)
    rec_mean  = recalls.mean() * 100
    rec_std   = recalls.std(ddof=1) * 100
    f1_mean   = f1s.mean() * 100
    f1_std    = f1s.std(ddof=1) * 100

    print("\n========== Cross-validation Results ==========")
    print(f"Gesture Accuracy: {acc_mean:.2f} ± {acc_std:.2f} %")
    print(f"Macro Recall:     {rec_mean:.2f} ± {rec_std:.2f} %")
    print(f"Macro F1:         {f1_mean:.2f} ± {f1_std:.2f} %")
    print(f"Validation Loss:  {loss_mean:.4f} ± {loss_std:.4f}")
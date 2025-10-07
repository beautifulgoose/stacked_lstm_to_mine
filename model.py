
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import List, Tuple, Optional, Dict, Any
from torch.optim.lr_scheduler import LambdaLR

class StackedLSTMClassifier(nn.Module):
    def __init__(
        self,
        # 兼容两种命名：input_dim / input_size（二选一必须给）
        input_dim: Optional[int] = None,
        *,
        input_size: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,           # 语义上要求≥2，这里我们手写两层 lstm1/lstm2
        bidirectional: bool = False,
        dropout: float = 0.1,
        num_classes: int = 29,
        use_attention_agg: bool = False,  # 是否使用注意力做“窗口聚合”（默认关）
    ):
        super().__init__()
        assert num_layers >= 2, "本实现默认两层以上（显式两层 LSTM）"

        # 统一 input_dim
        if input_dim is None and input_size is None:
            raise TypeError("必须提供 input_dim 或 input_size")
        if input_dim is None:
            input_dim = int(input_size)
        self.input_dim = int(input_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_attention_agg = use_attention_agg

        # 两层 LSTM（各自 num_layers=1）
        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,         # 我们内部统一用 (T,B,C)
            bidirectional=bidirectional,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * self.num_directions,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,
            bidirectional=bidirectional,
        )

        # Dropout：中间层（仅定长分支使用）+ 头部
        self.inter_drop = nn.Dropout(dropout)
        self.head_drop  = nn.Dropout(dropout)

        # 可选：注意力聚合（对“样本内的多个窗口”做加权）
        feat_dim = hidden_dim * self.num_directions
        if self.use_attention_agg:
            self.att = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.Tanh(),
                nn.Linear(feat_dim // 2, 1)   # 输出 (B,K,1) 的打分
            )

        # 分类头
        self.fc = nn.Linear(feat_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias" in name:
                    nn.init.constant_(p.data, 0.0)
            elif name.startswith("fc.weight"):
                nn.init.xavier_uniform_(p.data)
            elif name.startswith("fc.bias"):
                nn.init.constant_(p.data, 0.0)
            elif self.use_attention_agg and ("att.0.weight" in name or "att.2.weight" in name):
                nn.init.xavier_uniform_(p.data)
            elif self.use_attention_agg and ("att.0.bias" in name or "att.2.bias" in name):
                nn.init.constant_(p.data, 0.0)

    # ---- 将(B*K,T,C)的一批“窗口序列”编码成向量 (B*K, D*H) ----
    def _encode_window_seq(
            self,
            x_flat: torch.Tensor,  # (B*K, T, C)
            lengths_flat: Optional[torch.Tensor] = None  # (B*K,) 或 None
    ) -> torch.Tensor:
        DH = self.hidden_dim * self.num_directions

        if lengths_flat is not None:
            # 只编码有效窗口（len>0），无效窗口直接置零向量
            lengths_cpu = lengths_flat.to("cpu")
            valid_mask = lengths_cpu > 0  # (B*K,)
            feats_flat = x_flat.new_zeros(x_flat.size(0), DH)  # 预分配 (B*K, D*H)

            if valid_mask.any():
                x_valid = x_flat[valid_mask]  # (N_valid, T, C)
                TBT = x_valid.permute(1, 0, 2).contiguous()  # (T, N_valid, C)
                packed = nn.utils.rnn.pack_padded_sequence(
                    TBT, lengths_cpu[valid_mask], enforce_sorted=False
                )
                out1, _ = self.lstm1(packed)
                out2, _ = self.lstm2(out1)  # PackedSequence
                out2, _ = nn.utils.rnn.pad_packed_sequence(out2)  # (T, N_valid, D*H)
                feats_valid = out2.mean(dim=0)  # (N_valid, D*H)
                feats_flat[valid_mask] = feats_valid.to(feats_flat.dtype)

            feats_flat = self.head_drop(feats_flat)
            return feats_flat  # (B*K, D*H)

        else:
            # 定长/不提供长度：常规编码
            TBT = x_flat.permute(1, 0, 2).contiguous()  # (T, B*K, C)
            out1, _ = self.lstm1(TBT)  # (T, B*K, D*H)
            out1 = self.inter_drop(out1)
            out2, _ = self.lstm2(out1)  # (T, B*K, D*H)
            feats_flat = out2.mean(dim=0)  # (B*K, D*H)
            feats_flat = self.head_drop(feats_flat)
            return feats_flat                            # (B*K, D*H)

    def forward(
        self,
        x: torch.Tensor,                               # (B,T,C) 或 (B,K,T,C)
        lengths: Optional[torch.Tensor] = None,        # (B,) 或 (B,K,)
        win_mask: Optional[torch.Tensor] = None        # (B,K) 仅在 (B,K,T,C) 模式下需要
    ) -> torch.Tensor:
        # --- 情形A：窗口即样本，x=(B,T,C) ---
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.permute(1, 0, 2).contiguous()       # (T,B,C)

            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), enforce_sorted=False
                )
                out1, _ = self.lstm1(packed)
                out2, _ = self.lstm2(out1)
                out2, _ = nn.utils.rnn.pad_packed_sequence(out2)  # (T,B,D*H)
                feats = out2.mean(dim=0)                           # (B,D*H)
            else:
                out1, _ = self.lstm1(x)            # (T,B,D*H)
                out1 = self.inter_drop(out1)
                out2, _ = self.lstm2(out1)         # (T,B,D*H)
                feats = out2.mean(dim=0)           # (B,D*H)

            feats = self.head_drop(feats)
            logits = self.fc(feats)                # (B,num_classes)
            return logits

        # --- 情形B：样本内多窗口，x=(B,K,T,C) ---
        elif x.dim() == 4:
            B, K, T, C = x.shape
            x_flat = x.reshape(B * K, T, C)         # (B*K, T, C)

            lengths_flat = None
            if lengths is not None:
                lengths_flat = lengths.reshape(B * K)  # (B*K,)

            feats_flat = self._encode_window_seq(x_flat, lengths_flat)  # (B*K, D*H)
            feats = feats_flat.view(B, K, -1)                           # (B, K, D*H)

            # --- 聚合到样本级 ---
            if self.use_attention_agg:
                # 注意力分数：(B,K,1)
                scores = self.att(feats).squeeze(-1)                    # (B,K)
                if win_mask is not None:
                    scores = scores.masked_fill(win_mask == 0, -1e9)
                alpha = scores.softmax(dim=1)                           # (B,K)
                feats = (feats * alpha.unsqueeze(-1)).sum(dim=1)        # (B, D*H)
            else:
                if win_mask is not None:
                    mask = win_mask.float()                             # (B,K)
                    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                    feats = (feats * mask.unsqueeze(-1)).sum(dim=1) / denom
                else:
                    feats = feats.mean(dim=1)                           # (B, D*H)

            logits = self.fc(feats)                                     # (B, num_classes)
            return logits

        else:
            raise ValueError(f"期待 x 为 (B,T,C) 或 (B,K,T,C)，但得到 {tuple(x.shape)}")

# ========================================================================
# ==============================
# 1) 将单个 fold 的 dict 适配为 PyTorch Dataset
# ==============================

class IMUSampleDatasetGrouped(Dataset):
    """
    split_dict:
      X_imu: List[List[np.ndarray (L,F)]]  # 样本 -> 多个窗口
      y:     List[标签]
    """
    def __init__(self, split_dict, label_map_gesture):
        self.windows_per_sample = []
        self.labels = []
        for ws, y in zip(split_dict["X_imu"], split_dict["y"]):
            pack = []
            for w in ws:
                a = np.asarray(w[0] if isinstance(w, (list,tuple)) else w, dtype=np.float32)
                if a.ndim != 2:
                    raise ValueError(f"窗口必须是二维 (L,F)，got {a.shape}")
                if a.shape[0] < a.shape[1]:  # 若给的是 (F,L)
                    a = a.T
                pack.append(torch.from_numpy(a))
            self.windows_per_sample.append(pack)         # List[Tensor(L,F)]
            self.labels.append(label_map_gesture[str(y)])
        # 记录特征维
        self.feature_dim = self.windows_per_sample[0][0].shape[1]

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {"windows": self.windows_per_sample[idx], "y": self.labels[idx]}

def collate_group_pad(batch):
    """
    输入：每个元素是 {"windows": [Tensor(Li,F), ...], "y": int}
    输出：
      x:        (B, Kmax, Tmax, F)
      lengths:  (B, Kmax)       # 每个窗口的有效长度
      win_mask: (B, Kmax)       # 1=该位置有窗口, 0=padding
      y:        (B,)
    """
    B = len(batch)
    Kmax = max(len(item["windows"]) for item in batch)
    F = batch[0]["windows"][0].shape[1]
    Tmax = max(w.shape[0] for item in batch for w in item["windows"])

    x = torch.zeros(B, Kmax, Tmax, F, dtype=torch.float32)
    lengths = torch.zeros(B, Kmax, dtype=torch.long)
    win_mask = torch.zeros(B, Kmax, dtype=torch.bool)
    y = torch.tensor([item["y"] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        for j, w in enumerate(item["windows"]):
            L = w.shape[0]
            x[i, j, :L, :] = w
            lengths[i, j] = L
            win_mask[i, j] = True

    return x, lengths, win_mask, y


# ==============================
# 2) 标签映射工具：字符串 -> 索引
# ==============================
def build_label_map(values: list) -> dict[str, int]:
    """将标签（可能是字符串）映射为连续整数索引"""
    uniq = sorted({str(v) for v in values})
    return {k: i for i, k in enumerate(uniq)}

def _compute_macro_recall_f1_spe(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    """
    基于混淆矩阵计算 macro recall、macro F1、macro specificity（多分类）
    y_true, y_pred: (N,) LongTensor
    """
    with torch.no_grad():
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        tp = cm.diag()
        support = cm.sum(dim=1)     # TP + FN
        pred_pos = cm.sum(dim=0)    # TP + FP
        total = cm.sum()

        eps = 1e-12
        fn = support - tp
        fp = pred_pos - tp
        tn = total - tp - fp - fn

        recall_per_class = tp.float() / (support.float() + eps)
        precision_per_class = tp.float() / (pred_pos.float() + eps)
        specificity_per_class = tn.float() / (tn.float() + fp.float() + eps)
        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

        macro_recall = recall_per_class.mean().item()
        macro_f1 = f1_per_class.mean().item()
        macro_spe = specificity_per_class.mean().item()
        return macro_recall, macro_f1, macro_spe

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

        # —— DataLoader（样本级）——
        train_ds = IMUSampleDatasetGrouped(train_split, label_map_g)
        val_ds = IMUSampleDatasetGrouped(val_split, label_map_g)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=False, collate_fn=collate_group_pad)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                drop_last=False, collate_fn=collate_group_pad)

        # —— 模型 ——
        input_dim = 6
        # —— 模型 ——（把 29 改为 num_gestures）
        model = StackedLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            bidirectional=True,  # 建议：先开双向（见下）
            dropout=0.2,
            num_classes=num_gestures,  # ← 不要写死 29
            use_attention_agg=True,  # ← 建议打开注意力聚合
        ).to(device)

        # 优化器：Adam
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        def build_warmup_cosine(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.1):
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                # cosine from lr -> min_lr_ratio*lr
                t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                cos = 0.5 * (1 + np.cos(np.pi * t))
                return min_lr_ratio + (1 - min_lr_ratio) * cos

            return LambdaLR(optimizer, lr_lambda)
        # 创建 optimizer 后
        scheduler = build_warmup_cosine(optimizer, warmup_epochs=5, total_epochs=num_epochs)

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
            # ===================== 训练 =====================
            model.train()
            total_loss, n = 0.0, 0
            for batch in train_loader:
                # 兼容两种 collate 格式
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    x, lengths, win_mask, y_g = batch
                    x = x.to(device);
                    lengths = lengths.to(device);
                    win_mask = win_mask.to(device);
                    y_g = y_g.to(device)
                    logits_g = model(x, lengths=lengths, win_mask=win_mask)  # 样本级：B×K×T×C
                    bsz = x.size(0)
                else:
                    x, y_g = batch
                    x = x.to(device);
                    y_g = y_g.to(device)
                    logits_g = model(x)  # 窗口级：B×T×C
                    bsz = x.size(0)

                optimizer.zero_grad()
                loss = criterion(logits_g, y_g)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item() * bsz
                n += bsz
            train_loss = total_loss / max(1, n)

            # ===================== 验证 =====================
            model.eval()
            val_loss, correct_g, tot = 0.0, 0, 0
            all_true, all_pred = [], []

            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 4:
                        x, lengths, win_mask, y_g = batch
                        x = x.to(device);
                        lengths = lengths.to(device);
                        win_mask = win_mask.to(device);
                        y_g = y_g.to(device)
                        lg = model(x, lengths=lengths, win_mask=win_mask)
                        bsz = x.size(0)
                    else:
                        x, y_g = batch
                        x = x.to(device);
                        y_g = y_g.to(device)
                        lg = model(x)
                        bsz = x.size(0)

                    loss = criterion(lg, y_g)
                    val_loss += loss.item() * bsz

                    pred_g = lg.argmax(1)
                    correct_g += (pred_g == y_g).sum().item()
                    tot += bsz

                    all_true.append(y_g.detach().cpu())
                    all_pred.append(pred_g.detach().cpu())

            val_loss /= max(1, tot)
            acc_g = correct_g / max(1, tot)

            # 计算 macro recall & macro F1（注意这里用训练集构建的 label_map_g 大小）
            all_true = torch.cat(all_true, dim=0)
            all_pred = torch.cat(all_pred, dim=0)
            macro_recall, macro_f1, macro_spe = _compute_macro_recall_f1_spe(all_true, all_pred, num_classes=num_gestures)

            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
                  f"| val_loss={val_loss:.4f} | gesture_acc={acc_g:.3f}"
                  f"| macro_recall={macro_recall:.3f} | macro_f1={macro_f1:.3f} | macro_spe={macro_spe:.3f}")

            if acc_g > best_acc + min_delta:
                best_acc = acc_g
                best_val_loss = val_loss
                best_recall = macro_recall
                best_f1 = macro_f1
                best_spe = macro_spe
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if early_stopping and epochs_no_improve >= patience:
                print(f"早停触发！在第 {epoch} 轮停止，最佳 acc={best_acc:.3f}")
                break

        # —— 恢复并保存最佳模型 ——
        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(model.state_dict(), f"best_model_fold{fold_id}.pt")
            print(f"Fold {fold_id} 最优模型已保存 (val_acc={best_acc:.3f})")

        fold_results.append((best_acc, best_val_loss, best_recall, best_f1, best_spe))

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
    results = np.array(results)  # shape: (folds, 5)
    accs = results[:, 0]
    losses = results[:, 1]
    recalls = results[:, 2]
    f1s = results[:, 3]
    spes = results[:, 4]

    acc_mean = accs.mean() * 100
    acc_std = accs.std(ddof=1) * 100
    loss_mean = losses.mean()
    loss_std = losses.std(ddof=1)
    rec_mean = recalls.mean() * 100
    rec_std = recalls.std(ddof=1) * 100
    f1_mean = f1s.mean() * 100
    f1_std = f1s.std(ddof=1) * 100
    spe_mean = spes.mean() * 100
    spe_std = spes.std(ddof=1) * 100

    print("\n========== Cross-validation Results ==========")
    print(f"Gesture Accuracy: {acc_mean:.2f} ± {acc_std:.2f} %")
    print(f"Macro Recall:     {rec_mean:.2f} ± {rec_std:.2f} %")
    print(f"Macro F1:         {f1_mean:.2f} ± {f1_std:.2f} %")
    print(f"Macro Specificity:{spe_mean:.2f} ± {spe_std:.2f} %")
    print(f"Validation Loss:  {loss_mean:.4f} ± {loss_std:.4f}")

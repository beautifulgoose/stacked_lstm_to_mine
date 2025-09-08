import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import ROOT_DIR, ACCEL_GYRO_RATE
from data_processing import align_sensors, median_filter, window_process

def set_partition(selected_categories,collections):
    # 根据选择的动作类别构建文件夹路径
    selected_folders = []
    for collection in collections:
        collection_path = os.path.join(ROOT_DIR, collection)
        for category in selected_categories:
            category_path = os.path.join(collection_path, category)
            if os.path.exists(category_path):
                selected_folders.append(category_path)

    return selected_folders

# ------------------------- 加载和数据处理模块 -------------------------
def load_data(collection_name,action_dir):
    """
    返回独立的IMU,并确保窗口对齐
    """
    #数据加载load
    data = {}
    for file in os.listdir(action_dir):
        file_path = os.path.join(action_dir, file)
        if 'Accelerometer' in file:
            data['accel'] = pd.read_csv(file_path, usecols=[2, 3, 4, 5], header=None)
        elif 'Gyroscope' in file:
            data['gyro'] = pd.read_csv(file_path, usecols=[2, 3, 4, 5], header=None)

    #数据对齐
    aligned_imu = {}
    # 对齐IMU数据（加速度计+陀螺仪）
    if  data['accel'] is not None or data['gyro'] is not None:
        try:
            aligned_imu = align_sensors(
                accel_df=data['accel'],
                gyro_df=data['gyro'],
            )
        except ValueError as e:
            print(f"IMU alignment failed: {str(e)}")
            aligned_imu = {}

    return aligned_imu

# 数据处理process
def process_data(aligned_imu):
    windows_imu = []
    for sample in aligned_imu:
        # ==================== ★ 3. 低通滤波 (2-25 Hz) ★ ====================
        for key in ['accel', 'gyro']:
            sample[key] = median_filter(sample[key])

        window_imu= window_process(sample)
        windows_imu.append(window_imu)

    return windows_imu

# 针对训练集和测试集的文件检索与加载函数
def build_set_partition_dataset(selected_folders):
    X_imu, y, y_user = [], [], []
    for collection_path in selected_folders:

        collection_name = os.path.basename(collection_path)
        print(f"Processing collection: {collection_name}")

        for action_name in os.listdir(collection_path):
            if action_name == 'gesture_release':
                continue
            print(f"    Processing category: {action_name}")
            category_path = os.path.join(collection_path, action_name)
            if os.path.isdir(category_path):
                for device_name in os.listdir(category_path):
                    device_path = os.path.join(category_path, device_name)
                    if os.path.isdir(device_path):
                        if os.path.isdir(device_path):
                            for trial_name in os.listdir(device_path):
                                trial_path = os.path.join(device_path, trial_name)
                                if os.path.isdir(trial_path):
                                    # 获取特征及有效性标志
                                    imu_trial= load_data(
                                        collection_name, trial_path
                                    )

                                    X_imu.append(imu_trial)
                                    y.append(action_name)

    lengths = np.array([len(sample) for sample in X_imu], dtype=np.int32)
    return X_imu, np.array(y), lengths

# ------------------------- 交互：选择动作类别 -------------------------
def choose_categories():
    """让用户一次性选择动作类别，返回 selected_categories, collections"""
    # 读取所有的采集号
    collections = [d for d in os.listdir(ROOT_DIR)
                   if os.path.isdir(os.path.join(ROOT_DIR, d))]
    if not collections:
        raise ValueError("No collection folders found in ROOT_DIR")

    # 将所有的采集号和根目录拼接，形成访问路径
    action_categories = set()
    for collection in collections:
        collection_path = os.path.join(ROOT_DIR, collection)
        # 明确采集号下的动作类别
        cats = [d for d in os.listdir(collection_path)
                if os.path.isdir(os.path.join(collection_path, d))]
        action_categories.update(cats)
    if not action_categories:
        raise ValueError("No action categories found in collections")

    # 本实验固定使用1，4，5，6号的动作类别，此处写死
    print("Available action categories:")
    action_categories = sorted(action_categories)
    for i, cat in enumerate(action_categories, 1):
        print(f"{i}: {cat}")
    #selection = input("Enter category selection (e.g., 13 for first and third): ")
    selection = ["1","4","5","6"]
    print(selection)

    selected_categories = [action_categories[int(i) - 1]
                           for i in selection if i.isdigit()
                           and 1 <= int(i) <= len(action_categories)]
    if not selected_categories:
        raise ValueError("No category selected.")
    return selected_categories, collections
# -------------------------------------------------------------------


# ----------------------建立五折交叉验证训练测试集------------------------
def build_kfold_datasets(n_splits=5):
    """
    返回 datasets, selected_categories
    datasets: List[dict]，长度 = n_splits
              每个元素包含 'train' 和 'val' 两个子字典
    """
    selected_categories, collections = choose_categories()
    collections = np.array(collections)
    if len(collections) < n_splits:
        raise ValueError("Number of collection folders must ≥ n_splits.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    datasets = []


    for fold, (train_idx, val_idx) in enumerate(kf.split(collections), 1):
        print(f"\n===== Fold {fold}/{n_splits} =====")
        train_cols = collections[train_idx]
        val_cols = collections[val_idx]

        #构建读取路径
        train_folders = set_partition(selected_categories, train_cols)
        val_folders   = set_partition(selected_categories, val_cols)

        #构建训练集和测试集
        train_X, train_y, train_len = build_set_partition_dataset(train_folders)
        val_X,   val_y, val_len   = build_set_partition_dataset(val_folders)

        #数据处理(针对train_X和val_X)
        train_X_imu = process_data(train_X)
        val_X_imu = process_data(val_X)

        datasets.append({
            'train': {
                'X_imu': train_X_imu,
                'y': train_y,
            },
            'val': {
                'X_imu': val_X_imu,
                'y': val_y,
            }
        })
    return datasets, selected_categories
# -------------------------------------------------------------------
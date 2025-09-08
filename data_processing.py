import numpy as np
from numpy.lib._stride_tricks_impl import sliding_window_view
from scipy.signal import butter, filtfilt

from config import WINDOW_SECONDS, ACCEL_GYRO_RATE, OVERLAP_RATIO

#==============对齐函数=================
def align_sensors(accel_df=None, gyro_df=None):
    """返回分离的IMU和PPG对齐数据"""
    # 初始化返回结构
    aligned_imu = {}

    if accel_df is not None:
        accel_df.columns = ["timestamp", "x", "y", "z"]
    if gyro_df is not None:
        gyro_df.columns = ["timestamp", "x", "y", "z"]

    n = min(len(accel_df), len(gyro_df))

    if accel_df is not None:
        aligned_imu['accel'] = accel_df[:n][['x', 'y', 'z']].values

    if gyro_df is not None:
        aligned_imu['gyro'] = gyro_df[:n][['x', 'y', 'z']].values

    return aligned_imu

# ==================== 数据预处理 ====================
def median_filter(signal: np.ndarray, L: int = 3) -> np.ndarray:
    """
    对三通道加速度/陀螺数据做中值滤波（不改变通道数）。
    要求 signal 形状为 (T, 3)，沿时间维 axis=0 滤波。
    """
    if L % 2 == 0:
        raise ValueError("窗口长度 L 必须是奇数")
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] != 3:
        raise ValueError(f"期望 (T,3)，拿到 {signal.shape}")

    N = L // 2
    # 仅在时间维做边缘复制填充
    padded = np.pad(signal, pad_width=((N, N), (0, 0)), mode='edge')  # (T+2N, 3)
    # 生成滑动窗口: (T, L, 3)
    win = sliding_window_view(padded, window_shape=L, axis=0)
    # 沿窗口维取中值 -> (T, 3)
    med = np.median(win, axis=1)
    return med.astype(np.float32, copy=False)


def window_process(aligned_imu):
    window_size_imu = int(WINDOW_SECONDS * ACCEL_GYRO_RATE)
    step = int(window_size_imu * (1 - OVERLAP_RATIO))
    n_samples = len(next(iter(aligned_imu.values())))

    accel = aligned_imu['accel']
    gyro  = aligned_imu['gyro']

    # -------- 4. 滑动窗口裁剪 --------
    window_imu = []
    for i in range(0, n_samples - window_size_imu + 1, step):
        acc_window = accel[i: i + window_size_imu]      # (L,3)
        gyr_window = gyro[i: i + window_size_imu]   # (L,3)
        imu6 = np.concatenate([acc_window, gyr_window], axis=-1)  # (L,6)
        window_imu.append(imu6.astype(np.float32))

    return window_imu
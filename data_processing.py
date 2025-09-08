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
def lowpass_filter(data, cutoff=25, fs=100, order=4):
    # 低通滤波器，去除高频噪声，截止频率25hz
    normal_cutoff = cutoff / fs
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


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
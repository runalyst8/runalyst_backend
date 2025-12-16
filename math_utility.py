import numpy as np
from scipy.signal import savgol_filter, find_peaks


def normalize_arr(arr: np.ndarray) -> np.ndarray:
    # [1,J,3] -> [J,3]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr

def get_dt(times_list):
    return np.median(np.diff(times_list)) if len(times_list) > 2 else 1 / 30

def smooth(y_list, dts):
    win = int(max(5, round(0.20 / dts)))
    if win % 2 == 0:
        win += 1
    if win >= len(y_list):
        win = len(y_list) - 1 if (len(y_list) - 1) % 2 == 1 else len(y_list) - 2
        win = max(win, 5)
    y_s = savgol_filter(y_list, window_length=win, polyorder=3)
    return y_s

def find_y_peaks(ys_smooth, dts):
    min_dist = int(max(1, round(0.25 / dts)))
    prom = float(np.std(ys_smooth) * 0.3)
    peaks, props = find_peaks(ys_smooth, distance=min_dist, prominence=prom)
    return peaks

def find_minima_indices(ys_smooth, dts):
    min_dist = int(max(1, round(0.25 / dts)))
    prom = float(np.std(ys_smooth) * 0.3)
    minima, props = find_peaks(-ys_smooth, distance=min_dist, prominence=prom)
    return minima
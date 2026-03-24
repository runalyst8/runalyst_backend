import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

JSONL_PATH = "/Users/edd/Downloads/output_mid_strike_ceren.jsonl"

with open("joint_indices.json", "r") as f:
    joint_index = json.load(f)

RIGHT_ANKLE_INDEX = joint_index["right_ankle"]
RIGHT_FOOT_INDEX = joint_index["right_foot"]

timestamps = []
right_ankle_y = []
right_foot_y = []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        if "joints_3d" not in data or data["joints_3d"] is None:
            continue

        joints = data["joints_3d"]

        # Flatten [1,1,24,3] → [24,3]
        try:
            joints = joints[0][0]
        except:
            continue

        if len(joints) <= max(RIGHT_ANKLE_INDEX, RIGHT_FOOT_INDEX):
            continue

        ts = data.get("timestamp_sec")
        if ts is None:
            continue

        ankle_y = -1 * joints[RIGHT_ANKLE_INDEX][1]
        foot_y  = -1 * joints[RIGHT_FOOT_INDEX][1]

        timestamps.append(ts)
        right_ankle_y.append(ankle_y)
        right_foot_y.append(foot_y)

timestamps = np.array(timestamps)
right_ankle_y = np.array(right_ankle_y)
right_foot_y = np.array(right_foot_y)

# ---- Savitzky–Golay smoothing ----

# Estimate FPS
if len(timestamps) > 1:
    dt = np.mean(np.diff(timestamps))
    fps_est = 1.0 / dt
else:
    fps_est = 30

# Window ≈ 0.25 seconds
window_length = int(0.25 * fps_est)

if window_length % 2 == 0:
    window_length += 1

window_length = max(window_length, 5)

if window_length >= len(right_ankle_y):
    window_length = len(right_ankle_y) - 1
    if window_length % 2 == 0:
        window_length -= 1

poly_order = 3

ankle_smooth = savgol_filter(right_ankle_y, window_length, poly_order)
foot_smooth  = savgol_filter(right_foot_y, window_length, poly_order)

# ---- Plot ----
# ---- Plot ----

fig, axes = plt.subplots(2, 1, figsize=(12,10))

frame_idx = np.arange(len(right_ankle_y))

# -------- Graph 1: Time vs Y --------
axes[0].plot(timestamps, right_ankle_y, color="blue", alpha=0.25)
axes[0].plot(timestamps, right_foot_y, color="green", alpha=0.25)

axes[0].plot(timestamps, ankle_smooth, color="blue", label="Right Ankle (smoothed)")
axes[0].plot(timestamps, foot_smooth, color="green", label="Right Foot (smoothed)")

axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Flipped 3D Y Coordinate")
axes[0].set_title("Right Ankle vs Right Foot Y (Time)")
axes[0].legend()
axes[0].grid(True)

# -------- Graph 2: Frame Index vs Y --------
axes[1].plot(frame_idx, right_ankle_y, color="blue", alpha=0.25)
axes[1].plot(frame_idx, right_foot_y, color="green", alpha=0.25)

axes[1].plot(frame_idx, ankle_smooth, color="blue", label="Right Ankle (smoothed)")
axes[1].plot(frame_idx, foot_smooth, color="green", label="Right Foot (smoothed)")

axes[1].set_xlabel("Frame Index")
axes[1].set_ylabel("Flipped 3D Y Coordinate")
axes[1].set_title("Right Ankle vs Right Foot Y (Frame Index)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
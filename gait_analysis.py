import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

JSONL_PATH = "run_nlf.jsonl"

# İstediğin kaynak:
TARGET_PATH = "nlf.joints3d[0]"          # <-- bunu kullan
# TARGET_PATH = "nlf.joints3d_nonparam[0]"  # alternatif

JOINT_IDX = 23   # sağ ayak için deneyeceğin index (0-23 arası)
OUT_PLOT = f"joints3d_idx{JOINT_IDX}.png"

def normalize_arr(arr: np.ndarray) -> np.ndarray:
    # [1,J,3] -> [J,3]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr

def extract_by_path(nlf_obj, path: str):
    # path: "nlf.joints3d[0]" gibi
    p = path
    if p.startswith("nlf."):
        p = p[4:]
    cur = nlf_obj
    for part in p.split("."):
        if not part:
            continue
        if "[" in part:
            key, idx = part.split("[", 1)
            idx = int(idx.replace("]", ""))
            cur = cur[key][idx]
        else:
            cur = cur[part]
    return cur

records = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("ok") is True and "nlf" in rec:
            records.append(rec)

if len(records) == 0:
    raise SystemExit("JSONL içinde ok=true olan kayıt bulunamadı.")

times, ys = [], []

for rec in records:
    nlf = rec["nlf"]
    t = rec.get("timestamp_sec", None)
    if t is None:
        continue

    try:
        raw = extract_by_path(nlf, TARGET_PATH)
        arr = normalize_arr(np.array(raw, dtype=np.float32))

        # Beklenen: [24,3]
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        if not (0 <= JOINT_IDX < arr.shape[0]):
            raise SystemExit(f"JOINT_IDX={JOINT_IDX} out of range. This joints array has J={arr.shape[0]}.")

        y = float(arr[JOINT_IDX, 1])
    except Exception:
        continue

    times.append(float(t))
    ys.append(y)

times = np.array(times, dtype=np.float32)
y = np.array(ys, dtype=np.float32)

if len(y) < 10:
    raise SystemExit("Not enough valid frames after extraction. TARGET_PATH veya json formatı uyuşmuyor olabilir.")

# Smooth
dt = np.median(np.diff(times)) if len(times) > 2 else 1/30
win = int(max(5, round(0.20 / dt)))
if win % 2 == 0:
    win += 1
if win >= len(y):
    win = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
    win = max(win, 5)

y_s = savgol_filter(y, window_length=win, polyorder=3)

# Contact detection
# Sen "y ters" dedin -> contact ~ local MAX diyelim:
min_dist = int(max(1, round(0.25 / dt)))
prom = float(np.std(y_s) * 0.3)
peaks, props = find_peaks(y_s, distance=min_dist, prominence=prom)

# Eğer sende contact minimum çıkıyorsa, bunu kullan:
# peaks, props = find_peaks(-y_s, distance=min_dist, prominence=prom)

print(f"Using TARGET_PATH: {TARGET_PATH}  (J={arr.shape[0]} from last valid frame)")
print(f"Valid frames: {len(y)}")
print(f"Detected contact candidates: {len(peaks)}")
print("First 10 peaks (t, y_s):")
for i in peaks[:10]:
    print(f"t={times[i]:.3f}s  y={y_s[i]:.4f}")

cycles = [(times[peaks[k]], times[peaks[k+1]]) for k in range(len(peaks)-1)]
print(f"Total cycles: {len(cycles)}")
print("First 5 cycles (t_start -> t_end):")
for c in cycles[:5]:
    print(c)

# Plot
plt.figure()
plt.plot(times, y, label="raw y")
plt.plot(times, y_s, label="smoothed y")
plt.scatter(times[peaks], y_s[peaks], label="contact peaks", zorder=3)
plt.xlabel("time (s)")
plt.ylabel("y")
plt.title(f"{TARGET_PATH}, JOINT_IDX={JOINT_IDX}")
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PLOT, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", OUT_PLOT)

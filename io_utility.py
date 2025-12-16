import json
from math_utility import *
import matplotlib.pyplot as plt

JOINT_MAP = {
    "left_heel": 7,
    "right_heel": 8,
    "left_toe": 10,
    "right_toe": 11
}

def extract_by_path(nlf_obj, target_path: str):
    # path: "nlf.joints3d[0]" gibi
    p = target_path
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

def open_nlf_output_json(json_path: str):
    result = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("ok") is True and "nlf" in rec:
                result.append(rec)

    if len(result) == 0:
        raise SystemExit("JSONL içinde ok=true olan kayıt bulunamadı.")
    return result

def extract_ys_and_times(video_nlf, target_path, joint_idx):
    if isinstance(joint_idx, str):
        if joint_idx not in JOINT_MAP:
            raise ValueError(f"Unknown joint name '{joint_idx}'. Valid names: {list(JOINT_MAP.keys())}")
        joint_idx = JOINT_MAP[joint_idx]
    times, ys = [], []
    for frame_nlf in video_nlf:
        frame_nlf_data = frame_nlf["nlf"]
        t = frame_nlf.get("timestamp_sec", None)
        if t is None:
            continue

        try:
            joints_positions = extract_by_path(nlf_obj=frame_nlf_data, target_path=target_path)
            normalized_js_pos = normalize_arr(np.array(joints_positions, dtype=np.float32))

            # Beklenen: [24,3]
            if normalized_js_pos.ndim != 2 or normalized_js_pos.shape[1] != 3:
                continue
            if not (0 <= joint_idx < normalized_js_pos.shape[0]):
                raise SystemExit(f"JOINT_IDX={joint_idx} out of range. This joints array has J={normalized_js_pos.shape[0]}.")

            y = float(normalized_js_pos[joint_idx, 1])
        except Exception:
            continue

        times.append(float(t))
        ys.append(y)

    times = np.array(times, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)

    if len(y) < 10:
        raise SystemExit("Not enough valid frames after extraction. TARGET_PATH veya json formatı uyuşmuyor olabilir.")

    return times, y

def plot_results(times, ys, ys_smooth, indices, joint_idx, target_path):
    out_plot = f"joints3d_idx{joint_idx}.png"
    plt.figure()
    plt.plot(times, ys, label="raw y")
    plt.plot(times, ys_smooth, label="smoothed y")
    plt.scatter(times[indices], ys_smooth[indices], label="contact indices", zorder=3)
    plt.xlabel("time (s)")
    plt.ylabel("y")
    plt.title(f"{target_path}, JOINT_IDX={joint_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved:", out_plot)
import numpy as np
import argparse
import json
from contact_and_overstride import run_gait_pipeline

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]

def get_joint(frame, joint_name):
    idx = SMPL_JOINT_NAMES.index(joint_name)
    return np.array(frame["joints_3d"][0][0][idx], dtype=float)


def calculate_overstride_at_contact(
    frames: list,
    contacts: list,
    k: float = 0.7
) -> dict:
    """
    Calculate overstride index at contact frames.

    Parameters
    ----------
    frames : list
        List of frame dictionaries
    contacts : list
        List of contact dictionaries from gait pipeline
    k : float
        Weight for lean forward component (default 0.7)

    Returns
    -------
    dict with per_contact data and summary statistics
    """
    output = []

    for c in contacts:
        frame_idx = c["frame"]
        side = c["side"]
        frame = frames[frame_idx]

        if side == "L":
            foot = get_joint(frame, "left_foot")
            ankle = get_joint(frame, "left_ankle")
        else:
            foot = get_joint(frame, "right_foot")
            ankle = get_joint(frame, "right_ankle")

        pelvis = get_joint(frame, "pelvis")
        head = get_joint(frame, "head")

        # contact bölgesi merkezi
        contact_mid = (foot + ankle) / 2.0

        # alpha: pelvis -> contact_mid açısı (x-y sagittal düzlem)
        x = contact_mid[0] - pelvis[0]
        y = contact_mid[1] - pelvis[1]  # y aşağı doğru artıyorsa
        alpha = np.degrees(np.arctan2(x, y))

        # lean forward: pelvis -> head doğrusu, x-y düzleminde
        dx = head[0] - pelvis[0]
        dy = pelvis[1] - head[1]
        lean_forward = np.degrees(np.arctan2(dx, dy))

        # birleşik skor
        overstride_index = alpha - k * lean_forward

        output.append({
            "frame": int(frame_idx),
            "side": side,
            "alpha_deg": float(alpha),
            "lean_forward_deg": float(lean_forward),
            "overstride_index_deg": float(overstride_index),
            "x": float(x),
            "y": float(y)
        })

    if output:
        mean_alpha = float(np.mean([o["alpha_deg"] for o in output]))
        mean_lean = float(np.mean([o["lean_forward_deg"] for o in output]))
        mean_osi = float(np.mean([o["overstride_index_deg"] for o in output]))

        if mean_osi >= 10:
            comment = "Belirgin overstride eğilimi."
        elif mean_osi >= 5:
            comment = "Hafif overstride eğilimi."
        else:
            comment = "Stride kabul edilebilir / iyi."
    else:
        mean_alpha = None
        mean_lean = None
        mean_osi = None
        comment = "Contact bulunamadı."

    return {
        "per_contact": output,
        "mean_alpha_deg": mean_alpha,
        "mean_lean_forward_deg": mean_lean,
        "mean_overstride_index_deg": mean_osi,
        "comment": comment
    }


def main():
    parser = argparse.ArgumentParser(
        description="Contact frame'lerinde alpha, lean ve overstride index hesaplama."
    )
    parser.add_argument("jsonl_file")
    parser.add_argument("--label", default="Runner")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--k", type=float, default=0.7,
                        help="lean forward ağırlığı")
    args = parser.parse_args()

    result = run_gait_pipeline(
        path=args.jsonl_file,
        label=args.label,
        fps=args.fps,
        verbose=False,
        plot=False
    )

    frames = result["frames"]
    contacts = result["contacts"]

    final_result = calculate_overstride_at_contact(frames, contacts, k=args.k)

    print(json.dumps(final_result, indent=2, ensure_ascii=False))

    print("\n--- Özet ---")
    if final_result["mean_alpha_deg"] is not None:
        print(f"Ortalama alpha: {final_result['mean_alpha_deg']:.2f} derece")
        print(f"Ortalama lean forward: {final_result['mean_lean_forward_deg']:.2f} derece")
        print(f"Ortalama overstride index: {final_result['mean_overstride_index_deg']:.2f} derece")
    print(f"Yorum: {final_result['comment']}")


if __name__ == "__main__":
    main()
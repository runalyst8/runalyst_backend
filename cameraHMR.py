"""
Run CameraHMR on a video and save SMPL params + full mesh for each frame.

⚠️ IMPORTANT SETUP (before running):
1. Clone and install CameraHMR:
   git clone https://github.com/pixelite1201/CameraHMR.git
   cd CameraHMR
   pip install -r requirements.txt
   pip install smplx opencv-python

2. Download the SMPL neutral model from:
   https://smpl.is.tue.mpg.de/download.php
   and put the file here (for example):
   CameraHMR/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

3. Set the correct paths in the CONFIG section below:
   - VIDEO_PATH
   - OUTPUT_NPZ_PATH
   - HMR_CHECKPOINT_PATH
   - SMPL_MODEL_DIR

4. Run this script from inside the CameraHMR repo:
   python run_video_camerahmr.py
"""

import os
import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

# ---------- CONFIG ----------
VIDEO_PATH = "input.mp4"                      # path to your input video
OUTPUT_NPZ_PATH = "smpl_output_sequence.npz"  # where to save the npz
HMR_CHECKPOINT_PATH = "checkpoints/model_checkpoint.ckpt"  # adjust to your .ckpt path
SMPL_MODEL_DIR = "models"                     # folder containing SMPL .pkl
IMAGE_SIZE = 224                              # CameraHMR/HMR-style input size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

# --------- IMPORT HMR + SMPL ---------
from models.hmr import HMR    # this assumes CameraHMR has a HMR-like model here
from smplx import SMPL


def load_hmr_model(checkpoint_path: str, device: str = DEVICE) -> torch.nn.Module:
    """Load CameraHMR/HMR model from checkpoint."""
    print(f"[INFO] Loading HMR model from: {checkpoint_path}")
    model = HMR().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try common keys used in lightning-style checkpoints
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Remove 'module.' prefix if model was saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("[INFO] HMR model loaded.")
    return model


def load_smpl_model(model_dir: str, device: str = DEVICE) -> SMPL:
    """Load SMPL (neutral) model using smplx."""
    print(f"[INFO] Loading SMPL model from dir: {model_dir}")
    smpl_model = SMPL(
        model_path=model_dir,  # directory containing the SMPL .pkl
        gender="NEUTRAL",
        batch_size=1,
    ).to(device)
    print("[INFO] SMPL model loaded.")
    return smpl_model


def get_preprocess_transform() -> transforms.Compose:
    """Image preprocessing: resize + to tensor + (optional) normalize."""
    # NOTE: adjust normalization to match CameraHMR training if needed
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # If repo uses ImageNet normalization, uncomment:
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
    ])


def run_camerahmr_on_frame(
    frame_bgr: np.ndarray,
    model: torch.nn.Module,
    smpl_model: SMPL,
    preprocess: transforms.Compose,
    device: str = DEVICE,
):
    """
    Run CameraHMR on a single BGR frame and return:
    - pose: (1, 72)
    - betas: (1, 10)
    - cam: (1, ?)
    - verts: (1, 6890, 3)
    """

    # BGR -> RGB -> PIL
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Preprocess
    inp = preprocess(pil_img).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        out = model(inp)

    # Handle different possible output formats
    if isinstance(out, dict):
        # Try common keys; adjust if repo uses different names
        pred_pose = (
            out.get("pred_pose")
            or out.get("pose")
            or out.get("smpl_pose")
        )
        pred_betas = (
            out.get("pred_shape")
            or out.get("shape")
            or out.get("smpl_betas")
        )
        pred_cam = (
            out.get("pred_cam")
            or out.get("cam")
            or out.get("camera")
        )
        if pred_pose is None or pred_betas is None:
            raise RuntimeError(
                f"Model output dict keys not recognized: {out.keys()}"
            )
    else:
        # Tuple/list style: assume (pose, betas, cam)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            pred_pose = out[0]
            pred_betas = out[1]
            pred_cam = out[2] if len(out) > 2 else None
        else:
            raise RuntimeError(
                "Unexpected model output format. Got type: "
                f"{type(out)}; value: {out}"
            )

    # Run SMPL to get vertices
    # pred_pose: (B, 72) = [global_orient(3), body_pose(69)]
    body_pose = pred_pose[:, 3:]
    global_orient = pred_pose[:, :3]

    smpl_out = smpl_model(
        betas=pred_betas,
        body_pose=body_pose,
        global_orient=global_orient,
    )

    verts = smpl_out.vertices  # (B, 6890, 3)

    return pred_pose, pred_betas, pred_cam, verts


def main():
    # ---- Load models ----
    model = load_hmr_model(HMR_CHECKPOINT_PATH, DEVICE)
    smpl_model = load_smpl_model(SMPL_MODEL_DIR, DEVICE)
    preprocess = get_preprocess_transform()

    # ---- Open video ----
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    all_poses = []
    all_betas = []
    all_cams = []
    all_verts = []

    frame_id = 0
    print(f"[INFO] Processing video: {VIDEO_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"[INFO] Processing frame {frame_id}...")
        try:
            pred_pose, pred_betas, pred_cam, verts = run_camerahmr_on_frame(
                frame_bgr=frame,
                model=model,
                smpl_model=smpl_model,
                preprocess=preprocess,
                device=DEVICE,
            )
        except Exception as e:
            print(f"[WARN] Skipping frame {frame_id} due to error: {e}")
            frame_id += 1
            continue

        # Move to CPU and remove batch dimension
        all_poses.append(pred_pose.cpu().numpy()[0])  # (72,)
        all_betas.append(pred_betas.cpu().numpy()[0])  # (10,)
        if pred_cam is not None:
            all_cams.append(pred_cam.cpu().numpy()[0])  # (?,)
        else:
            all_cams.append(None)
        all_verts.append(verts.cpu().numpy()[0])      # (6890, 3)

        frame_id += 1

    cap.release()
    print(f"[INFO] Done. Processed {frame_id} frames.")

    if frame_id == 0:
        raise RuntimeError("No frames processed; check your video file.")

    # ---- Stack results ----
    poses = np.stack(all_poses, axis=0)   # (N, 72)
    betas = np.stack(all_betas, axis=0)   # (N, 10)
    verts = np.stack(all_verts, axis=0)   # (N, 6890, 3)

    # Cameras may be None; handle that
    if all_cams[0] is not None:
        cams = np.stack(all_cams, axis=0)
    else:
        cams = None

    # ---- Save to NPZ ----
    print(f"[INFO] Saving results to: {OUTPUT_NPZ_PATH}")
    if cams is not None:
        np.savez(
            OUTPUT_NPZ_PATH,
            poses=poses,
            betas=betas,
            verts=verts,
            cams=cams,
        )
    else:
        np.savez(
            OUTPUT_NPZ_PATH,
            poses=poses,
            betas=betas,
            verts=verts,
        )
    print("[INFO] Saved NPZ with keys:", list(np.load(OUTPUT_NPZ_PATH).keys()))


if __name__ == "__main__":
    main()

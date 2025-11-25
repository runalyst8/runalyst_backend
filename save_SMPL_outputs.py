"""
NLF GAIT → SMPL PARAMS + FULL MESH EXPORT
-----------------------------------------

This script:

1. Loads the NLF TorchScript model
2. Loads a video
3. Extracts frames
4. Runs NLF on each frame
5. Collects SMPL parameters (pose, betas, trans)
6. Collects SMPL mesh vertices
7. Saves everything into a single NPZ file
8. Saves mesh frames as OBJ files for Blender/MeshLab

You ONLY need:
- nlf_l_multi.torchscript
- SMPL model folder (for faces)
"""

import os
import cv2
import torch
import torchvision
import numpy as np
from smplx import SMPL


# -------------------------------------------------------
# 1. PATHS AND CONFIG
# -------------------------------------------------------
VIDEO_PATH = "run.mp4"               # Change this to your video file
NLF_MODEL_PATH = "models/nlf_l_multi.torchscript"
SMPL_MODEL_PATH = "models/SMPL"      # Folder containing SMPL model files
FRAME_DIR = "frames"
OBJ_DIR = "mesh_objs"
NPZ_OUTPUT_PATH = "smpl_output_sequence.npz"


# -------------------------------------------------------
# 2. LOAD NLF TORCHSCRIPT MODEL
# -------------------------------------------------------
print("Loading NLF model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.jit.load(NLF_MODEL_PATH).to(device).eval()
print("Model loaded. Using:", device)


# -------------------------------------------------------
# 3. EXTRACT FRAMES FROM VIDEO
# -------------------------------------------------------
def extract_frames(video_path, out_dir="frames", step=1):
    """
    Extract frames from a video.
    step=1 means every frame; step=2 means every other frame.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            fp = f"{out_dir}/frame_{idx:05d}.jpg"
            cv2.imwrite(fp, frame)
            frame_paths.append(fp)

        idx += 1

    cap.release()
    print(f"Extracted {len(frame_paths)} frames.")
    return frame_paths


print("Extracting video frames...")
frames = extract_frames(VIDEO_PATH, FRAME_DIR)


# -------------------------------------------------------
# 4. RUN NLF ON ALL FRAMES
# -------------------------------------------------------
print("Running NLF on frames...")

results = []
for fp in frames:
    # NLF expects uint8 CHW format
    img = torchvision.io.read_image(fp).to(device)
    batch = img.unsqueeze(0)  # shape (1,3,H,W)

    with torch.inference_mode():
        pred = model.detect_smpl_batched(batch)

    results.append(pred)

print("Inference complete on all frames.")


# -------------------------------------------------------
# 5. COLLECT SMPL PARAMS AND MESH VERTICES
# -------------------------------------------------------
print("Collecting SMPL parameters and mesh vertices...")

poses = []   # (T, 72)
betas = []   # (T, 10)
trans = []   # (T, 3)
verts = []   # (T, V, 3)

for pred in results:
    poses.append(pred["pose"][0].cpu().numpy())   # (72,)
    betas.append(pred["betas"][0].cpu().numpy())  # (10,)
    trans.append(pred["trans"][0].cpu().numpy())  # (3,)
    verts.append(pred["verts"][0].cpu().numpy())  # (6890,3)

poses = np.stack(poses, axis=0)
betas = np.stack(betas, axis=0)
trans = np.stack(trans, axis=0)
verts = np.stack(verts, axis=0)

print("poses:", poses.shape)
print("betas:", betas.shape)
print("trans:", trans.shape)
print("verts:", verts.shape)


# -------------------------------------------------------
# 6. SAVE EVERYTHING INTO A SINGLE NPZ FILE
# -------------------------------------------------------
print("Saving full SMPL sequence to:", NPZ_OUTPUT_PATH)

np.savez(
    NPZ_OUTPUT_PATH,
    poses=poses,  
    betas=betas,
    trans=trans,
    verts=verts,
)

print("NPZ file saved.")


# -------------------------------------------------------
# 7. SAVE PER-FRAME OBJ MESHES
# -------------------------------------------------------
print("Loading SMPL model to get faces...")

smpl_model = SMPL(model_path=SMPL_MODEL_PATH, gender="NEUTRAL")
faces = smpl_model.faces   # (F, 3)

def save_obj(path, vertices, faces):
    """Write a mesh to OBJ format."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

os.makedirs(OBJ_DIR, exist_ok=True)

print("Saving OBJ meshes...")

for t in range(len(verts)):
    out_path = f"{OBJ_DIR}/frame_{t:05d}.obj"
    save_obj(out_path, verts[t], faces)

print(f"Saved {len(verts)} OBJ files to {OBJ_DIR}/")

print("\nAll done!")

#!/usr/bin/env python3
import argparse
import json
import time
from typing import Optional, Dict, Any

import cv2
import requests
from tqdm import tqdm


def post_frame(
    session: requests.Session,
    url: str,
    jpeg_bytes: bytes,
    timeout: float,
    retries: int,
    backoff: float,
) -> Dict[str, Any]:
    """
    POST /analyze_frame with multipart file upload, return JSON dict.
    Retries on transient failures.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            files = {
                "file": ("frame.jpg", jpeg_bytes, "image/jpeg")
            }
            r = session.post(url, files=files, timeout=timeout)
            if r.status_code == 200:
                return r.json()

            # 4xx genelde kalıcıdır (input/validation), 5xx retry mantıklı olabilir.
            if 400 <= r.status_code < 500:
                raise RuntimeError(f"Client error {r.status_code}: {r.text[:300]}")
            raise RuntimeError(f"Server error {r.status_code}: {r.text[:300]}")

        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep_s = backoff * (2 ** attempt)
                time.sleep(sleep_s)
            else:
                break

    raise RuntimeError(f"Failed after retries. Last error: {last_err}") from last_err


def main():
    ap = argparse.ArgumentParser(description="Extract NLF params per frame via FastAPI /analyze_frame")
    ap.add_argument("--video", required=True, help="Input video path (mp4/mov/etc.)")
    ap.add_argument("--api", default="http://127.0.0.1:8080/analyze_frame", help="Analyze endpoint URL")
    ap.add_argument("--out", required=True, help="Output JSONL path, e.g. out_nlf.jsonl")
    ap.add_argument("--every-n", type=int, default=1, help="Process every Nth frame (default: 1 = all frames)")
    ap.add_argument("--start", type=int, default=0, help="Start frame index (default: 0)")
    ap.add_argument("--end", type=int, default=-1, help="End frame index inclusive (default: -1 = to end)")
    ap.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (1-100)")
    ap.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=2, help="Retry count on failure")
    ap.add_argument("--backoff", type=float, default=0.6, help="Backoff base seconds (exponential)")
    ap.add_argument("--max-frames", type=int, default=-1, help="Process at most this many frames (default: -1 no limit)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start = max(args.start, 0)
    end = args.end if args.end >= 0 else (total - 1 if total > 0 else -1)
    if end >= 0 and end < start:
        raise SystemExit("--end must be >= --start (or -1).")

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # JPEG encode params
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    session = requests.Session()

    processed = 0
    to_process_est = None
    if total > 0 and end >= 0:
        span = (end - start + 1)
        to_process_est = (span + args.every_n - 1) // args.every_n

    pbar = tqdm(total=to_process_est, desc="Frames", unit="frame")

    with open(args.out, "w", encoding="utf-8") as f:
        frame_idx = start
        while True:
            if end >= 0 and frame_idx > end:
                break
            if args.max_frames >= 0 and processed >= args.max_frames:
                break

            ok, frame_bgr = cap.read()
            if not ok:
                break

            if (frame_idx - start) % args.every_n != 0:
                frame_idx += 1
                continue

            # timestamp
            ts = (frame_idx / fps) if fps and fps > 0 else None

            # Encode to JPEG bytes (BGR -> JPEG)
            ok2, buf = cv2.imencode(".jpg", frame_bgr, encode_params)
            if not ok2:
                # write an error record and continue
                rec = {
                    "frame_index": frame_idx,
                    "timestamp_sec": ts,
                    "ok": False,
                    "error": "jpeg_encode_failed",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                frame_idx += 1
                pbar.update(1)
                continue

            jpeg_bytes = buf.tobytes()

            # Call API
            try:
                pred = post_frame(
                    session=session,
                    url=args.api,
                    jpeg_bytes=jpeg_bytes,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff=args.backoff,
                )
                rec = {
                    "frame_index": frame_idx,
                    "timestamp_sec": ts,
                    "ok": True,
                    "nlf": pred,
                }
            except Exception as e:
                rec = {
                    "frame_index": frame_idx,
                    "timestamp_sec": ts,
                    "ok": False,
                    "error": str(e),
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            processed += 1
            frame_idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Done. Wrote: {args.out}")


if __name__ == "__main__":
    main()

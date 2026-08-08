from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--frames", nargs="+", type=int)
    selection.add_argument("--times", nargs="+", type=float)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    saved = []
    if args.frames is not None:
        requests = [("frame", frame_index) for frame_index in args.frames]
    else:
        requests = [("time_seconds", time_seconds) for time_seconds in args.times]

    for request_type, value in requests:
        if value < 0:
            saved.append(
                {
                    request_type: value,
                    "saved": False,
                    "reason": f"{request_type} cannot be negative",
                }
            )
            continue
        if request_type == "frame":
            cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            output_name = f"frame_{int(value):05d}.png"
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(value) * 1000.0)
            output_name = f"time_{float(value):010.6f}.png"
        ok, frame = cap.read()
        if not ok:
            saved.append(
                {
                    request_type: value,
                    "saved": False,
                    "reason": "frame could not be read",
                }
            )
            continue
        output_path = out_dir / output_name
        if not cv2.imwrite(str(output_path), frame):
            saved.append(
                {
                    request_type: value,
                    "saved": False,
                    "reason": "image could not be written",
                }
            )
            continue
        saved.append(
            {
                request_type: value,
                "decoded_frame_position": int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
                "decoded_time_seconds": float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0,
                "saved": True,
                "path": str(output_path),
            }
        )

    cap.release()

    summary = {"video": str(video_path), "saved_frames": saved}
    (out_dir / "frame_extract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

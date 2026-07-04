from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    saved = []
    for frame_index in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            saved.append({"frame": frame_index, "saved": False, "reason": "frame could not be read"})
            continue
        output_path = out_dir / f"frame_{frame_index:05d}.png"
        cv2.imwrite(str(output_path), frame)
        saved.append({"frame": frame_index, "saved": True, "path": str(output_path)})

    cap.release()

    summary = {"video": str(video_path), "saved_frames": saved}
    (out_dir / "frame_extract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.video_io import get_video_info, require_cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=300)
    parser.add_argument("--point-radius", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_info = get_video_info(args.video)
    gaze = load_gaze_txt(args.gaze, video_width=video_info.width, video_height=video_info.height)
    valid_gaze = gaze.dataframe[gaze.dataframe["valid"]].copy()
    valid_gaze = valid_gaze.dropna(subset=["time_seconds"]).sort_values("time_seconds")
    if valid_gaze.empty:
        raise SystemExit("No valid gaze samples with rec_time were found.")
    gaze_times = valid_gaze["time_seconds"].to_numpy()
    gaze_x = valid_gaze["gaze_x"].to_numpy()
    gaze_y = valid_gaze["gaze_y"].to_numpy()
    cv2 = require_cv2()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    out_path = out_dir / "gaze_overlay_sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, video_info.fps, (video_info.width, video_info.height))

    frame_index = args.start_frame
    written = 0
    matched_points = 0
    while frame_index < args.end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame_time = frame_index / video_info.fps if video_info.fps else 0.0
        idx = int(gaze_times.searchsorted(frame_time, side="left"))
        if idx >= len(gaze_times):
            idx = len(gaze_times) - 1
        if idx > 0 and abs(gaze_times[idx - 1] - frame_time) < abs(gaze_times[idx] - frame_time):
            idx -= 1
        if abs(gaze_times[idx] - frame_time) <= 0.05:
            cv2.circle(
                frame,
                (int(round(gaze_x[idx])), int(round(gaze_y[idx]))),
                args.point_radius,
                (0, 0, 255),
                -1,
            )
            matched_points += 1
        writer.write(frame)
        frame_index += 1
        written += 1

    cap.release()
    writer.release()

    summary = {
        "output_video": str(out_path),
        "written_frames": written,
        "matched_points": matched_points,
        "matching_method": "nearest valid gaze sample by rec_time",
        "video_info": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "frame_count": video_info.frame_count,
        },
        "gaze_summary": summarize_gaze(gaze),
    }
    (out_dir / "gaze_overlay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

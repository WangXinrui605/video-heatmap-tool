from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.heatmap import (
    add_brush,
    build_final_heatmap,
    normalize_with_max,
    overlay_heatmap,
)
from video_heatmap.time_range import resolve_selected_range
from video_heatmap.video_io import get_video_info, require_cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--brush-size", type=int, default=40)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args()
    started_at = perf_counter()

    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be 1 or higher.")

    cv2 = require_cv2()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_info = get_video_info(args.video)
    try:
        selected_range = resolve_selected_range(
            video_info,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    gaze = load_gaze_txt(args.gaze, video_width=video_info.width, video_height=video_info.height)
    valid_gaze = gaze.dataframe[gaze.dataframe["valid"]].copy()
    valid_gaze = valid_gaze.dropna(subset=["time_seconds"])
    selected = valid_gaze[
        (valid_gaze["time_seconds"] >= selected_range.start_time_seconds)
        & (valid_gaze["time_seconds"] <= selected_range.end_time_seconds)
    ]
    if selected.empty:
        raise SystemExit("No valid gaze samples were found in the selected frame range.")

    selected = selected.sort_values("time_seconds")
    gaze_times = selected["time_seconds"].to_numpy(dtype=float)
    gaze_points = selected[["gaze_x", "gaze_y"]].to_numpy(dtype=float)

    # Use the final maximum for all frames so the heatmap colours do not rescale every frame.
    final_heatmap = build_final_heatmap(gaze_points, video_info.width, video_info.height, args.brush_size)
    final_max = float(np.nanmax(final_heatmap)) if final_heatmap.size else 0.0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_range.start_frame)

    output_video_path = out_dir / "heatmap_development.mp4"
    summary_path = out_dir / "heatmap_development_summary.json"
    output_fps = video_info.fps / args.frame_skip
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        output_fps,
        (video_info.width, video_info.height),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {output_video_path}")

    heatmap = np.zeros((video_info.height, video_info.width), dtype=np.float32)
    gaze_index = 0
    written_frames = 0
    read_frames = 0
    added_gaze_samples = 0
    frame_index = selected_range.start_frame

    while frame_index < selected_range.end_frame_exclusive:
        should_write_frame = (frame_index - selected_range.start_frame) % args.frame_skip == 0
        if should_write_frame:
            ok, frame = cap.read()
        else:
            ok = cap.grab()
            frame = None
        if not ok:
            break

        frame_time = frame_index / video_info.fps
        while gaze_index < len(gaze_times) and gaze_times[gaze_index] <= frame_time:
            x, y = gaze_points[gaze_index]
            add_brush(heatmap, x, y, args.brush_size)
            gaze_index += 1
            added_gaze_samples += 1

        if should_write_frame:
            heatmap_uint8 = normalize_with_max(heatmap, final_max)
            blended, _ = overlay_heatmap(frame, heatmap_uint8, args.opacity)
            writer.write(blended)
            written_frames += 1

        read_frames += 1
        frame_index += 1

    cap.release()
    writer.release()

    summary = {
        "output_video": str(output_video_path),
        "written_frames": written_frames,
        "read_frames": read_frames,
        "added_gaze_samples": added_gaze_samples,
        "mode": "cumulative",
        "selected_range": selected_range.as_dict(),
        "parameters": {
            "brush_size": args.brush_size,
            "opacity": args.opacity,
            "frame_skip": args.frame_skip,
            "output_fps": output_fps,
        },
        "video_info": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "frame_count": video_info.frame_count,
        },
        "gaze_summary": summarize_gaze(gaze),
        "selected_valid_gaze_samples": int(len(selected)),
        "runtime_seconds": perf_counter() - started_at,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

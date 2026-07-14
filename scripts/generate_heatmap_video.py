from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.heatmap import add_brush, build_final_heatmap
from video_heatmap.video_io import get_video_info, require_cv2


def normalize_with_max(heatmap: np.ndarray, max_value: float) -> np.ndarray:
    if max_value <= 0:
        return np.zeros_like(heatmap, dtype=np.uint8)
    return np.clip((heatmap / max_value) * 255.0, 0, 255).astype(np.uint8)


def overlay_heatmap(background, heatmap_uint8, opacity: float):
    cv2 = require_cv2()
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    alpha = (heatmap_uint8.astype("float32") / 255.0) * float(opacity)
    alpha = alpha[:, :, None]
    blended = background.astype("float32") * (1.0 - alpha) + color_heatmap.astype("float32") * alpha
    return blended.clip(0, 255).astype("uint8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--brush-size", type=int, default=40)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args()

    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be 1 or higher.")

    cv2 = require_cv2()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_info = get_video_info(args.video)
    end_frame = args.end_frame if args.end_frame is not None else video_info.frame_count
    end_frame = min(end_frame, video_info.frame_count)
    if args.start_frame < 0 or args.start_frame >= end_frame:
        raise SystemExit("Selected frame range is not valid.")

    start_time = args.start_frame / video_info.fps
    last_output_frame = end_frame - 1
    end_time = last_output_frame / video_info.fps

    gaze = load_gaze_txt(args.gaze, video_width=video_info.width, video_height=video_info.height)
    valid_gaze = gaze.dataframe[gaze.dataframe["valid"]].copy()
    valid_gaze = valid_gaze.dropna(subset=["time_seconds"])
    selected = valid_gaze[(valid_gaze["time_seconds"] >= start_time) & (valid_gaze["time_seconds"] <= end_time)]
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

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
    frame_index = args.start_frame

    while frame_index < end_frame:
        should_write_frame = (frame_index - args.start_frame) % args.frame_skip == 0
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
            writer.write(overlay_heatmap(frame, heatmap_uint8, args.opacity))
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
        "selected_range": {
            "start_frame": args.start_frame,
            "end_frame_exclusive": end_frame,
            "last_output_frame": last_output_frame,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
        },
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
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.heatmap import build_final_heatmap, overlay_heatmap
from video_heatmap.rendering import HeatmapGradient, rendering_summary, resolve_heat_scale, scale_heatmap
from video_heatmap.time_range import resolve_selected_range
from video_heatmap.video_io import get_video_info, require_cv2


def read_background_frame(video_path: str, frame_index: int):
    cv2 = require_cv2()
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


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
    parser.add_argument("--gradient", choices=[item.value for item in HeatmapGradient], default="jet")
    parser.add_argument("--heat-scale-mode", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--heat-scale-maximum", type=float)
    args = parser.parse_args()
    started_at = perf_counter()

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

    points = selected[["gaze_x", "gaze_y"]].to_numpy()
    heatmap = build_final_heatmap(points, video_info.width, video_info.height, args.brush_size)
    try:
        heat_scale = resolve_heat_scale(args.heat_scale_mode, args.heat_scale_maximum)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    heatmap_uint8, resolved_maximum = scale_heatmap(heatmap, heat_scale)

    background_frame = read_background_frame(args.video, selected_range.start_frame)
    overlay, color_heatmap = overlay_heatmap(background_frame, heatmap_uint8, args.opacity, args.gradient)

    heatmap_gray_path = out_dir / "final_heatmap_gray.png"
    heatmap_color_path = out_dir / "final_heatmap_color.png"
    overlay_path = out_dir / "final_heatmap_overlay.png"
    summary_path = out_dir / "final_heatmap_summary.json"

    cv2.imwrite(str(heatmap_gray_path), heatmap_uint8)
    cv2.imwrite(str(heatmap_color_path), color_heatmap)
    cv2.imwrite(str(overlay_path), overlay)

    summary = {
        "outputs": {
            "final_heatmap_gray": str(heatmap_gray_path),
            "final_heatmap_color": str(heatmap_color_path),
            "final_heatmap_overlay": str(overlay_path),
        },
        "selected_range": selected_range.as_dict(),
        "parameters": {
            "brush_size": args.brush_size,
            "opacity": args.opacity,
            "rendering": rendering_summary(args.gradient, heat_scale, resolved_maximum=resolved_maximum),
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

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.heatmap import build_final_heatmap, normalize_to_uint8
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


def overlay_heatmap(background, heatmap_uint8, opacity: float):
    cv2 = require_cv2()
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    alpha = (heatmap_uint8.astype("float32") / 255.0) * float(opacity)
    alpha = alpha[:, :, None]
    blended = background.astype("float32") * (1.0 - alpha) + color_heatmap.astype("float32") * alpha
    return blended.clip(0, 255).astype("uint8"), color_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--brush-size", type=int, default=40)
    parser.add_argument("--opacity", type=float, default=0.55)
    args = parser.parse_args()

    cv2 = require_cv2()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_info = get_video_info(args.video)
    end_frame = args.end_frame if args.end_frame is not None else video_info.frame_count
    start_time = args.start_frame / video_info.fps
    end_time = end_frame / video_info.fps

    gaze = load_gaze_txt(args.gaze, video_width=video_info.width, video_height=video_info.height)
    valid_gaze = gaze.dataframe[gaze.dataframe["valid"]].copy()
    valid_gaze = valid_gaze.dropna(subset=["time_seconds"])
    selected = valid_gaze[(valid_gaze["time_seconds"] >= start_time) & (valid_gaze["time_seconds"] <= end_time)]
    if selected.empty:
        raise SystemExit("No valid gaze samples were found in the selected frame range.")

    points = selected[["gaze_x", "gaze_y"]].to_numpy()
    heatmap = build_final_heatmap(points, video_info.width, video_info.height, args.brush_size)
    heatmap_uint8 = normalize_to_uint8(heatmap)

    background_frame = read_background_frame(args.video, args.start_frame)
    overlay, color_heatmap = overlay_heatmap(background_frame, heatmap_uint8, args.opacity)

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
        "selected_range": {
            "start_frame": args.start_frame,
            "end_frame": end_frame,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
        },
        "parameters": {
            "brush_size": args.brush_size,
            "opacity": args.opacity,
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

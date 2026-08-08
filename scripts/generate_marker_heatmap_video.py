from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt
from video_heatmap.heatmap import add_brush, build_final_heatmap
from video_heatmap.marker import distance_to_marker_square, discover_marker_columns
from video_heatmap.video_io import get_video_info, require_cv2


MARKER_LOCAL_CORNERS = np.array(
    [
        [0.0, 0.0],  # P1
        [1.0, 0.0],  # P2
        [1.0, 1.0],  # P3
        [0.0, 1.0],  # P4
    ],
    dtype=np.float32,
)


def normalize_with_max(heatmap: np.ndarray, max_value: float) -> np.ndarray:
    if max_value <= 0:
        return np.zeros_like(heatmap, dtype=np.uint8)
    return np.clip((heatmap / max_value) * 255.0, 0, 255).astype(np.uint8)


def marker_corner_valid_mask(corners: np.ndarray, min_area: float) -> np.ndarray:
    finite = np.isfinite(corners).all(axis=(1, 2))
    nonzero = ~np.all(np.isclose(corners, 0.0), axis=(1, 2))
    x = corners[:, :, 0]
    y = corners[:, :, 1]
    area = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
    )
    return finite & nonzero & (area >= min_area)


def nearest_time_index(times: np.ndarray, target: float) -> int:
    idx = int(times.searchsorted(target, side="left"))
    if idx >= len(times):
        idx = len(times) - 1
    if idx > 0 and abs(times[idx - 1] - target) < abs(times[idx] - target):
        idx -= 1
    return idx


def summarize_values(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def draw_cross(cv2, frame: np.ndarray, point: np.ndarray, color) -> None:
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.line(frame, (x - 11, y), (x + 11, y), color, 3, cv2.LINE_AA)
    cv2.line(frame, (x, y - 11), (x, y + 11), color, 3, cv2.LINE_AA)


def overlay_warped_heatmap(
    cv2,
    frame: np.ndarray,
    local_heatmap_uint8: np.ndarray,
    canvas_to_screen: np.ndarray,
    opacity: float,
) -> np.ndarray:
    frame_size = (frame.shape[1], frame.shape[0])
    local_color = cv2.applyColorMap(local_heatmap_uint8, cv2.COLORMAP_JET)
    screen_color = cv2.warpPerspective(
        local_color,
        canvas_to_screen,
        frame_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    screen_intensity = cv2.warpPerspective(
        local_heatmap_uint8,
        canvas_to_screen,
        frame_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = (screen_intensity.astype(np.float32) / 255.0) * float(opacity)
    alpha = alpha[:, :, None]
    blended = (
        frame.astype(np.float32) * (1.0 - alpha)
        + screen_color.astype(np.float32) * alpha
    )
    return blended.clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a cumulative heatmap in one D-Lab marker coordinate "
            "system and project it back onto each Scene Cam frame."
        )
    )
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--marker", default="Brasilia")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=300)
    parser.add_argument("--brush-radius-marker", type=float, default=1.0)
    parser.add_argument("--canvas-pixels-per-marker", type=float, default=40.0)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--max-time-delta", type=float, default=0.05)
    parser.add_argument("--min-marker-area", type=float, default=100.0)
    parser.add_argument(
        "--max-marker-distance",
        type=float,
        default=None,
        help=(
            "Only accumulate gaze at most this many marker units from the "
            "[0,1] marker square. By default, distant extrapolated gaze is "
            "retained."
        ),
    )
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help=(
            "Draw marker corners, the latest newly accumulated screen gaze, "
            "its current-marker reprojection, and timing diagnostics."
        ),
    )
    args = parser.parse_args()

    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be 1 or higher.")
    if args.brush_radius_marker <= 0:
        raise SystemExit("--brush-radius-marker must be positive.")
    if args.canvas_pixels_per_marker <= 0:
        raise SystemExit("--canvas-pixels-per-marker must be positive.")
    if args.max_marker_distance is not None and args.max_marker_distance < 0:
        raise SystemExit("--max-marker-distance must be zero or positive.")
    if not 0.0 <= args.opacity <= 1.0:
        raise SystemExit("--opacity must be between 0 and 1.")

    video_info = get_video_info(args.video)
    end_frame = min(args.end_frame, video_info.frame_count)
    if args.start_frame < 0 or args.start_frame >= end_frame:
        raise SystemExit("Invalid frame range.")
    start_time = args.start_frame / video_info.fps
    end_time = (end_frame - 1) / video_info.fps

    gaze = load_gaze_txt(
        args.gaze,
        video_width=video_info.width,
        video_height=video_info.height,
    )
    dataframe = gaze.dataframe.copy()
    marker_map = {
        marker.name: marker
        for marker in discover_marker_columns(dataframe.columns)
        if marker.has_all_corners and marker.has_gaze_coordinates
    }
    if args.marker not in marker_map:
        available = ", ".join(sorted(marker_map)) or "none"
        raise SystemExit(
            f"Marker {args.marker!r} is unavailable. Available markers: {available}"
        )
    marker = marker_map[args.marker]

    corner_columns = [column for pair in marker.corner_cols for column in pair]
    corners = (
        dataframe[corner_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .reshape(-1, 4, 2)
    )
    marker_x = pd.to_numeric(
        dataframe[marker.gaze_x_col], errors="coerce"
    ).to_numpy(dtype=float)
    marker_y = pd.to_numeric(
        dataframe[marker.gaze_y_col], errors="coerce"
    ).to_numpy(dtype=float)
    times = dataframe["time_seconds"].to_numpy(dtype=float)
    valid_corners = marker_corner_valid_mask(corners, args.min_marker_area)

    # For this first validation prototype, only use rows where the exported
    # screen gaze, marker-relative gaze, and marker corners are all valid at
    # the same timestamp. This is conservative and avoids stale marker values.
    # Pandas may expose this column as a read-only NumPy view. The mask is
    # refined in place below, so request an independent writable copy.
    candidate_gaze_samples = dataframe["valid"].to_numpy(dtype=bool, copy=True)
    candidate_gaze_samples &= np.isfinite(times)
    candidate_gaze_samples &= np.isfinite(marker_x) & np.isfinite(marker_y)
    candidate_gaze_samples &= valid_corners
    candidate_gaze_samples &= (times >= start_time) & (times <= end_time)

    all_marker_points = np.column_stack((marker_x, marker_y))
    all_marker_distances = distance_to_marker_square(all_marker_points)
    candidate_marker_distances = all_marker_distances[candidate_gaze_samples]
    valid_gaze_samples = candidate_gaze_samples.copy()
    if args.max_marker_distance is not None:
        valid_gaze_samples &= all_marker_distances <= args.max_marker_distance

    gaze_times = times[valid_gaze_samples]
    gaze_points_local = np.column_stack(
        (marker_x[valid_gaze_samples], marker_y[valid_gaze_samples])
    )
    gaze_points_screen = np.column_stack(
        (
            dataframe.loc[valid_gaze_samples, "gaze_x"].to_numpy(dtype=float),
            dataframe.loc[valid_gaze_samples, "gaze_y"].to_numpy(dtype=float),
        )
    )
    order = np.argsort(gaze_times)
    gaze_times = gaze_times[order]
    gaze_points_local = gaze_points_local[order]
    gaze_points_screen = gaze_points_screen[order]
    selected_marker_distances = all_marker_distances[valid_gaze_samples][order]
    if gaze_points_local.size == 0:
        if args.max_marker_distance is None:
            detail = ""
        else:
            detail = (
                f" within --max-marker-distance={args.max_marker_distance:g}"
            )
        raise SystemExit(
            "No conservative marker-relative gaze samples were found" + detail + "."
        )

    marker_times = times[valid_corners & np.isfinite(times)]
    marker_corners = corners[valid_corners & np.isfinite(times)]
    marker_order = np.argsort(marker_times)
    marker_times = marker_times[marker_order]
    marker_corners = marker_corners[marker_order]

    margin = args.brush_radius_marker * 2.0
    local_min = np.min(gaze_points_local, axis=0) - margin
    local_max = np.max(gaze_points_local, axis=0) + margin
    local_span = np.maximum(local_max - local_min, 1.0)
    scale = float(args.canvas_pixels_per_marker)
    canvas_width = max(2, int(np.ceil(local_span[0] * scale)) + 1)
    canvas_height = max(2, int(np.ceil(local_span[1] * scale)) + 1)
    if canvas_width > 2400 or canvas_height > 2400:
        shrink = min(2400 / canvas_width, 2400 / canvas_height)
        scale *= shrink
        canvas_width = max(2, int(np.ceil(local_span[0] * scale)) + 1)
        canvas_height = max(2, int(np.ceil(local_span[1] * scale)) + 1)

    gaze_points_canvas = np.column_stack(
        (
            (gaze_points_local[:, 0] - local_min[0]) * scale,
            (gaze_points_local[:, 1] - local_min[1]) * scale,
        )
    )
    brush_radius_canvas = max(1, int(round(args.brush_radius_marker * scale)))
    final_heatmap = build_final_heatmap(
        gaze_points_canvas,
        width=canvas_width,
        height=canvas_height,
        radius=brush_radius_canvas,
    )
    final_max = float(np.nanmax(final_heatmap)) if final_heatmap.size else 0.0

    canvas_to_local = np.array(
        [
            [1.0 / scale, 0.0, local_min[0]],
            [0.0, 1.0 / scale, local_min[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    cv2 = require_cv2()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{args.marker.lower()}_marker_heatmap"
    if args.debug_overlay:
        output_stem += "_debug"
    output_path = out_dir / f"{output_stem}.mp4"

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    output_fps = video_info.fps / args.frame_skip
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (video_info.width, video_info.height),
    )
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise SystemExit("Could not open input video or output video writer.")

    heatmap = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    gaze_index = 0
    frame_index = args.start_frame
    written_frames = 0
    frames_with_marker_projection = 0
    debug_frames_with_recent_gaze = 0
    debug_compensated_displacements: list[float] = []
    debug_marker_time_deltas: list[float] = []
    debug_gaze_time_deltas: list[float] = []

    while frame_index < end_frame:
        should_write = (frame_index - args.start_frame) % args.frame_skip == 0
        if should_write:
            ok, frame = cap.read()
        else:
            ok = cap.grab()
            frame = None
        if not ok:
            break

        frame_time = frame_index / video_info.fps
        while gaze_index < len(gaze_times) and gaze_times[gaze_index] <= frame_time:
            x, y = gaze_points_canvas[gaze_index]
            add_brush(heatmap, x, y, brush_radius_canvas)
            gaze_index += 1

        if should_write:
            marker_index = nearest_time_index(marker_times, frame_time)
            marker_delta = abs(marker_times[marker_index] - frame_time)
            marker_available = marker_delta <= args.max_time_delta
            local_to_screen = None
            if marker_available:
                local_to_screen = cv2.getPerspectiveTransform(
                    MARKER_LOCAL_CORNERS,
                    marker_corners[marker_index].astype(np.float32),
                ).astype(np.float64)
                debug_marker_time_deltas.append(float(marker_delta))

            if local_to_screen is not None and gaze_index > 0:
                canvas_to_screen = local_to_screen @ canvas_to_local
                heatmap_uint8 = normalize_with_max(heatmap, final_max)
                frame = overlay_warped_heatmap(
                    cv2,
                    frame,
                    heatmap_uint8,
                    canvas_to_screen,
                    args.opacity,
                )
                frames_with_marker_projection += 1
                status = f"{args.marker} marker heatmap"
                status_color = (0, 255, 0)
            elif local_to_screen is not None:
                status = f"{args.marker} marker available | waiting for gaze"
                status_color = (0, 255, 0)
            else:
                status = f"{args.marker} marker unavailable"
                status_color = (0, 165, 255)

            if args.debug_overlay:
                if marker_available:
                    polygon = np.rint(marker_corners[marker_index]).astype(np.int32)
                    cv2.polylines(
                        frame, [polygon], True, (0, 255, 0), 3, cv2.LINE_AA
                    )
                    for corner_number, point in enumerate(polygon, start=1):
                        px, py = int(point[0]), int(point[1])
                        cv2.circle(
                            frame, (px, py), 5, (0, 255, 0), -1, cv2.LINE_AA
                        )
                        cv2.putText(
                            frame,
                            str(corner_number),
                            (px + 7, py - 7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                recent_gaze_index = gaze_index - 1
                recent_gaze_delta = (
                    frame_time - gaze_times[recent_gaze_index]
                    if recent_gaze_index >= 0
                    else float("inf")
                )
                debug_lines = [
                    f"frame={frame_index} time={frame_time:.3f}s",
                    f"marker dt={marker_delta * 1000.0:.1f}ms",
                ]
                if (
                    recent_gaze_index >= 0
                    and 0.0 <= recent_gaze_delta <= args.max_time_delta
                ):
                    local_point = gaze_points_local[recent_gaze_index]
                    screen_point = gaze_points_screen[recent_gaze_index]
                    debug_frames_with_recent_gaze += 1
                    debug_gaze_time_deltas.append(float(recent_gaze_delta))
                    cv2.circle(
                        frame,
                        (
                            int(round(screen_point[0])),
                            int(round(screen_point[1])),
                        ),
                        9,
                        (0, 0, 255),
                        -1,
                        cv2.LINE_AA,
                    )

                    debug_lines.append(
                        "gaze local="
                        f"({local_point[0]:.3f},{local_point[1]:.3f}) "
                        f"dt={recent_gaze_delta * 1000.0:.1f}ms"
                    )
                    debug_lines.append(
                        "distance from marker square="
                        f"{float(distance_to_marker_square(local_point)):.2f} units"
                    )

                    if local_to_screen is not None:
                        projected = cv2.perspectiveTransform(
                            local_point.astype(np.float32).reshape(1, 1, 2),
                            local_to_screen.astype(np.float32),
                        ).reshape(2)
                        if np.isfinite(projected).all():
                            compensated_displacement = float(
                                np.linalg.norm(projected - screen_point)
                            )
                            debug_compensated_displacements.append(
                                compensated_displacement
                            )
                            debug_lines.append(
                                "old-screen to compensated displacement="
                                f"{compensated_displacement:.2f}px"
                            )
                            projected_inside = (
                                0 <= projected[0] < video_info.width
                                and 0 <= projected[1] < video_info.height
                            )
                            if projected_inside:
                                draw_cross(cv2, frame, projected, (255, 255, 0))
                                cv2.line(
                                    frame,
                                    (
                                        int(round(screen_point[0])),
                                        int(round(screen_point[1])),
                                    ),
                                    (
                                        int(round(projected[0])),
                                        int(round(projected[1])),
                                    ),
                                    (255, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                            else:
                                debug_lines.append(
                                    "current-marker reprojection is off-screen"
                                )

                text_y = 78
                for line in debug_lines:
                    cv2.putText(
                        frame,
                        line,
                        (24, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    text_y += 28

            cv2.putText(
                frame,
                status,
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            written_frames += 1

        frame_index += 1

    cap.release()
    writer.release()

    summary = {
        "output_video": str(output_path),
        "mode": "cumulative marker-coordinate heatmap",
        "marker": args.marker,
        "frame_range": {
            "start_frame_inclusive": args.start_frame,
            "end_frame_exclusive": end_frame,
        },
        "selected_marker_gaze_samples": int(len(gaze_times)),
        "candidate_marker_gaze_samples_before_distance_filter": int(
            np.count_nonzero(candidate_gaze_samples)
        ),
        "distance_filtered_gaze_samples": int(
            np.count_nonzero(candidate_gaze_samples)
            - np.count_nonzero(valid_gaze_samples)
        ),
        "valid_marker_corner_samples": int(len(marker_times)),
        "written_frames": written_frames,
        "frames_with_marker_projection": frames_with_marker_projection,
        "final_heatmap_max": final_max,
        "local_bounds": {
            "x_min": float(local_min[0]),
            "x_max": float(local_max[0]),
            "y_min": float(local_min[1]),
            "y_max": float(local_max[1]),
        },
        "canvas": {
            "width": canvas_width,
            "height": canvas_height,
            "pixels_per_marker": scale,
            "brush_radius_pixels": brush_radius_canvas,
            "brush_radius_marker_units": args.brush_radius_marker,
        },
        "parameters": {
            "opacity": args.opacity,
            "max_time_delta_seconds": args.max_time_delta,
            "min_marker_area_pixels": args.min_marker_area,
            "max_marker_distance_units": args.max_marker_distance,
            "frame_skip": args.frame_skip,
            "output_fps": output_fps,
            "debug_overlay": args.debug_overlay,
        },
        "important_limitation": (
            "All selected gaze samples are expressed in the chosen marker "
            "plane. This prototype does not determine whether gaze was "
            "semantically on the participant, face, table, or background."
        ),
        "distance_from_marker_square_units": {
            "candidate_samples": summarize_values(
                candidate_marker_distances.tolist()
            ),
            "selected_samples": summarize_values(
                selected_marker_distances.tolist()
            ),
        },
    }
    if args.debug_overlay:
        summary["debug"] = {
            "legend": {
                "red_circle": "latest newly accumulated exported screen gaze",
                "cyan_cross": (
                    "the same marker-relative gaze projected through the "
                    "marker corners matched to the current video frame"
                ),
                "green_polygon": "marker corners matched to the current frame",
            },
            "frames_with_recent_gaze": debug_frames_with_recent_gaze,
            "marker_time_delta_seconds": summarize_values(
                debug_marker_time_deltas
            ),
            "gaze_time_delta_seconds": summarize_values(debug_gaze_time_deltas),
            "old_screen_to_compensated_displacement_pixels": summarize_values(
                debug_compensated_displacements
            ),
        }
    summary_path = out_dir / f"{output_stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

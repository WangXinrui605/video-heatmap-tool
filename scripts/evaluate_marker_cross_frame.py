from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt
from video_heatmap.marker import distance_to_marker_square, discover_marker_columns
from video_heatmap.video_io import get_video_info, require_cv2


MARKER_LOCAL_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
)
MARKER_COLORS = {
    "Brasilia": (0, 255, 0),
    "Seoul": (0, 255, 255),
}


def nearest_time_index(times: np.ndarray, target: float) -> int:
    index = int(times.searchsorted(target, side="left"))
    if index >= len(times):
        index = len(times) - 1
    if index > 0 and abs(times[index - 1] - target) < abs(times[index] - target):
        index -= 1
    return index


def marker_corner_valid_mask(corners: np.ndarray, min_area: float) -> np.ndarray:
    finite = np.isfinite(corners).all(axis=(1, 2))
    nonzero = ~np.all(np.isclose(corners, 0.0), axis=(1, 2))
    x = corners[:, :, 0]
    y = corners[:, :, 1]
    area = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
    )
    return finite & nonzero & (area >= min_area)


def summarize_values(values: np.ndarray, unit: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            f"mean_{unit}": None,
            f"median_{unit}": None,
            f"p95_{unit}": None,
            f"max_{unit}": None,
        }
    return {
        "count": int(values.size),
        f"mean_{unit}": float(np.mean(values)),
        f"median_{unit}": float(np.median(values)),
        f"p95_{unit}": float(np.quantile(values, 0.95)),
        f"max_{unit}": float(np.max(values)),
    }


def distance_band(distance: float, near_limit: float) -> str:
    if np.isclose(distance, 0.0):
        return "inside"
    if distance <= near_limit:
        return "near"
    return "far"


def summarize_availability(
    first_available: np.ndarray, second_available: np.ndarray
) -> dict:
    first = np.asarray(first_available, dtype=bool)
    second = np.asarray(second_available, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Availability arrays must have the same shape.")
    return {
        "evaluated_frames": int(first.size),
        "both_available_frames": int(np.count_nonzero(first & second)),
        "first_only_frames": int(np.count_nonzero(first & ~second)),
        "second_only_frames": int(np.count_nonzero(~first & second)),
        "neither_available_frames": int(np.count_nonzero(~first & ~second)),
        "first_dropout_frames": int(np.count_nonzero(~first)),
        "second_dropout_frames": int(np.count_nonzero(~second)),
    }


def project_points(cv2, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(shaped, transform).reshape(-1, 2).astype(float)


def numeric_marker_data(dataframe: pd.DataFrame, marker) -> dict:
    corner_columns = [column for pair in marker.corner_cols for column in pair]
    corners = (
        dataframe[corner_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .reshape(-1, 4, 2)
    )
    marker_x = pd.to_numeric(dataframe[marker.gaze_x_col], errors="coerce").to_numpy(
        dtype=float
    )
    marker_y = pd.to_numeric(dataframe[marker.gaze_y_col], errors="coerce").to_numpy(
        dtype=float
    )
    return {
        "corners": corners,
        "gaze": np.column_stack((marker_x, marker_y)),
    }


def match_marker_to_frames(
    times: np.ndarray,
    corners: np.ndarray,
    valid: np.ndarray,
    frame_times: np.ndarray,
    max_time_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_rows = np.flatnonzero(valid)
    matched_rows = np.full(frame_times.size, -1, dtype=int)
    available = np.zeros(frame_times.size, dtype=bool)
    if valid_rows.size == 0:
        return matched_rows, available
    valid_times = times[valid_rows]
    order = np.argsort(valid_times)
    valid_rows = valid_rows[order]
    valid_times = valid_times[order]
    for frame_offset, frame_time in enumerate(frame_times):
        sample_offset = nearest_time_index(valid_times, float(frame_time))
        if abs(valid_times[sample_offset] - frame_time) <= max_time_delta:
            matched_rows[frame_offset] = valid_rows[sample_offset]
            available[frame_offset] = True
    return matched_rows, available


def evaluate_direction(
    cv2,
    anchor_name: str,
    target_name: str,
    marker_data: dict,
    frame_indices: np.ndarray,
    frame_times: np.ndarray,
    matched_rows: dict,
    availability: dict,
) -> tuple[list[dict], dict]:
    both = availability[anchor_name] & availability[target_name]
    offsets = np.flatnonzero(both)
    if offsets.size == 0:
        return [], {
            "anchor_marker": anchor_name,
            "target_landmark": f"{target_name} marker centre",
            "count": 0,
            "reason": "No frames had both markers available.",
        }

    reference_offset = int(offsets[0])
    anchor_reference = marker_data[anchor_name]["corners"][
        matched_rows[anchor_name][reference_offset]
    ]
    target_reference = marker_data[target_name]["corners"][
        matched_rows[target_name][reference_offset]
    ].mean(axis=0)
    anchor_to_screen = cv2.getPerspectiveTransform(
        MARKER_LOCAL_CORNERS, anchor_reference.astype(np.float32)
    )
    screen_to_anchor = np.linalg.inv(anchor_to_screen)
    target_in_anchor = project_points(
        cv2, target_reference.reshape(1, 2), screen_to_anchor
    )[0]

    records = []
    for offset in offsets:
        anchor_corners = marker_data[anchor_name]["corners"][
            matched_rows[anchor_name][offset]
        ]
        target_corners = marker_data[target_name]["corners"][
            matched_rows[target_name][offset]
        ]
        transform = cv2.getPerspectiveTransform(
            MARKER_LOCAL_CORNERS, anchor_corners.astype(np.float32)
        )
        predicted = project_points(cv2, target_in_anchor.reshape(1, 2), transform)[0]
        observed = target_corners.mean(axis=0)
        records.append(
            {
                "anchor_marker": anchor_name,
                "target_marker": target_name,
                "reference_frame": int(frame_indices[reference_offset]),
                "frame": int(frame_indices[offset]),
                "time_seconds": float(frame_times[offset]),
                "landmark_local_x": float(target_in_anchor[0]),
                "landmark_local_y": float(target_in_anchor[1]),
                "predicted_x": float(predicted[0]),
                "predicted_y": float(predicted[1]),
                "observed_x": float(observed[0]),
                "observed_y": float(observed[1]),
                "error_pixels": float(np.linalg.norm(predicted - observed)),
            }
        )

    errors = np.array([record["error_pixels"] for record in records], dtype=float)
    summary = {
        "anchor_marker": anchor_name,
        "target_landmark": f"{target_name} marker centre",
        "reference_frame": int(frame_indices[reference_offset]),
        "reference_time_seconds": float(frame_times[reference_offset]),
        "landmark_in_anchor_coordinates": target_in_anchor.tolist(),
        "landmark_distance_from_anchor_square_units": float(
            distance_to_marker_square(target_in_anchor)
        ),
        **summarize_values(errors, "pixels"),
    }
    return records, summary


def gaze_motion_summary(
    cv2,
    marker_name: str,
    marker_data: dict,
    times: np.ndarray,
    valid_corner_rows: np.ndarray,
    valid_screen_gaze: np.ndarray,
    start_time: float,
    end_time: float,
    frame_period: float,
    max_time_delta: float,
    near_limit: float,
) -> dict:
    data = marker_data[marker_name]
    gaze = data["gaze"]
    candidate = valid_corner_rows.copy()
    candidate &= valid_screen_gaze
    candidate &= np.isfinite(times)
    candidate &= np.isfinite(gaze).all(axis=1)
    candidate &= (times >= start_time) & (times <= end_time)

    valid_corner_indices = np.flatnonzero(valid_corner_rows & np.isfinite(times))
    valid_corner_times = times[valid_corner_indices]
    order = np.argsort(valid_corner_times)
    valid_corner_indices = valid_corner_indices[order]
    valid_corner_times = valid_corner_times[order]

    distances = []
    displacements = []
    bands = []
    if valid_corner_times.size == 0:
        return {
            "definition": (
                "One-output-frame compensated displacement of the same marker-local "
                "gaze point; this is a motion-sensitivity measure, not accuracy."
            ),
            "distance_from_marker_square": summarize_values(
                np.array([]), "marker_units"
            ),
            "compensated_displacement": summarize_values(np.array([]), "pixels"),
            "distance_bands": {},
        }
    for row in np.flatnonzero(candidate):
        next_time = float(times[row] + frame_period)
        nearest_offset = nearest_time_index(valid_corner_times, next_time)
        if abs(valid_corner_times[nearest_offset] - next_time) > max_time_delta:
            continue
        next_row = int(valid_corner_indices[nearest_offset])
        current_transform = cv2.getPerspectiveTransform(
            MARKER_LOCAL_CORNERS, data["corners"][row].astype(np.float32)
        )
        next_transform = cv2.getPerspectiveTransform(
            MARKER_LOCAL_CORNERS, data["corners"][next_row].astype(np.float32)
        )
        point = gaze[row]
        current_screen = project_points(cv2, point.reshape(1, 2), current_transform)[0]
        next_screen = project_points(cv2, point.reshape(1, 2), next_transform)[0]
        distance = float(distance_to_marker_square(point))
        distances.append(distance)
        displacements.append(float(np.linalg.norm(next_screen - current_screen)))
        bands.append(distance_band(distance, near_limit))

    distances_array = np.asarray(distances, dtype=float)
    displacements_array = np.asarray(displacements, dtype=float)
    result = {
        "definition": (
            "One-output-frame compensated displacement of the same marker-local "
            "gaze point; this is a motion-sensitivity measure, not accuracy."
        ),
        "distance_from_marker_square": summarize_values(
            distances_array, "marker_units"
        ),
        "compensated_displacement": summarize_values(displacements_array, "pixels"),
        "distance_bands": {},
    }
    for band in ("inside", "near", "far"):
        selected = np.array([value == band for value in bands], dtype=bool)
        result["distance_bands"][band] = {
            "definition": (
                "distance = 0"
                if band == "inside"
                else (
                    f"0 < distance <= {near_limit:g}"
                    if band == "near"
                    else f"distance > {near_limit:g}"
                )
            ),
            "distance": summarize_values(distances_array[selected], "marker_units"),
            "compensated_displacement": summarize_values(
                displacements_array[selected], "pixels"
            ),
        }
    return result


def save_evidence_frames(
    cv2,
    video_path: str,
    out_dir: Path,
    records: list[dict],
    marker_data: dict,
    matched_rows: dict,
    start_frame: int,
) -> list[str]:
    if not records:
        return []
    by_direction: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        key = (record["anchor_marker"], record["target_marker"])
        by_direction.setdefault(key, []).append(record)

    selected: list[tuple[str, dict]] = []
    for direction_records in by_direction.values():
        ordered = sorted(direction_records, key=lambda item: item["error_pixels"])
        non_reference = [
            item
            for item in ordered
            if item["frame"] != item["reference_frame"]
        ]
        success = non_reference[0] if non_reference else ordered[0]
        selected.extend(
            [
                ("success", success),
                ("typical", ordered[len(ordered) // 2]),
                ("failure", ordered[-1]),
            ]
        )

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    saved = []
    for label, record in selected:
        frame_index = int(record["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        offset = frame_index - start_frame
        for marker_name, color in MARKER_COLORS.items():
            row = matched_rows[marker_name][offset]
            if row < 0:
                continue
            polygon = np.rint(marker_data[marker_name]["corners"][row]).astype(np.int32)
            cv2.polylines(frame, [polygon], True, color, 3, cv2.LINE_AA)
        predicted = (int(round(record["predicted_x"])), int(round(record["predicted_y"])))
        observed = (int(round(record["observed_x"])), int(round(record["observed_y"])))
        cv2.circle(frame, predicted, 12, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, observed, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(frame, predicted, observed, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(
            frame,
            (
                f"{record['anchor_marker']} predicts {record['target_marker']} "
                f"centre: {record['error_pixels']:.1f}px"
            ),
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        filename = (
            f"frame_{frame_index:05d}_{record['anchor_marker'].lower()}_to_"
            f"{record['target_marker'].lower()}_{label}.png"
        )
        output_path = frames_dir / filename
        cv2.imwrite(str(output_path), frame)
        saved.append(str(output_path))
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-frame validation of the two P2 clothing markers. A target marker "
            "centre is stored in the other marker's local coordinates at a reference "
            "frame, then reprojected and compared with the observed target centre."
        )
    )
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=300)
    parser.add_argument("--max-time-delta", type=float, default=0.05)
    parser.add_argument("--min-marker-area", type=float, default=100.0)
    parser.add_argument("--near-distance", type=float, default=2.0)
    args = parser.parse_args()

    if args.max_time_delta < 0 or args.min_marker_area < 0 or args.near_distance < 0:
        raise SystemExit("Time delta, marker area, and near distance must be non-negative.")

    video_info = get_video_info(args.video)
    end_frame = min(args.end_frame, video_info.frame_count)
    if args.start_frame < 0 or args.start_frame >= end_frame:
        raise SystemExit("Invalid frame range.")
    frame_indices = np.arange(args.start_frame, end_frame, dtype=int)
    frame_times = frame_indices / video_info.fps

    gaze_result = load_gaze_txt(
        args.gaze, video_width=video_info.width, video_height=video_info.height
    )
    dataframe = gaze_result.dataframe
    marker_map = {
        marker.name: marker
        for marker in discover_marker_columns(dataframe.columns)
        if marker.has_all_corners and marker.has_gaze_coordinates
    }
    required = ("Brasilia", "Seoul")
    missing = [name for name in required if name not in marker_map]
    if missing:
        raise SystemExit(
            "Cross-frame two-marker validation requires Brasilia and Seoul; missing: "
            + ", ".join(missing)
        )

    times = dataframe["time_seconds"].to_numpy(dtype=float)
    screen_valid = dataframe["valid"].to_numpy(dtype=bool)
    marker_data = {
        name: numeric_marker_data(dataframe, marker_map[name]) for name in required
    }
    valid_corner_rows = {
        name: marker_corner_valid_mask(
            marker_data[name]["corners"], args.min_marker_area
        )
        & np.isfinite(times)
        for name in required
    }
    matched_rows = {}
    availability = {}
    for name in required:
        matched_rows[name], availability[name] = match_marker_to_frames(
            times,
            marker_data[name]["corners"],
            valid_corner_rows[name],
            frame_times,
            args.max_time_delta,
        )

    cv2 = require_cv2()
    records = []
    direction_summaries = []
    for anchor_name, target_name in (("Brasilia", "Seoul"), ("Seoul", "Brasilia")):
        direction_records, direction_summary = evaluate_direction(
            cv2,
            anchor_name,
            target_name,
            marker_data,
            frame_indices,
            frame_times,
            matched_rows,
            availability,
        )
        records.extend(direction_records)
        direction_summaries.append(direction_summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "marker_cross_frame_samples.csv"
    fieldnames = [
        "anchor_marker",
        "target_marker",
        "reference_frame",
        "frame",
        "time_seconds",
        "landmark_local_x",
        "landmark_local_y",
        "predicted_x",
        "predicted_y",
        "observed_x",
        "observed_y",
        "error_pixels",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    evidence_frames = save_evidence_frames(
        cv2,
        args.video,
        out_dir,
        records,
        marker_data,
        matched_rows,
        args.start_frame,
    )
    availability_summary = summarize_availability(
        availability["Brasilia"], availability["Seoul"]
    )
    availability_summary["first_marker"] = "Brasilia"
    availability_summary["second_marker"] = "Seoul"
    per_marker_availability = {
        name: {
            "available_frames": int(np.count_nonzero(availability[name])),
            "dropout_frames": int(np.count_nonzero(~availability[name])),
        }
        for name in required
    }
    gaze_motion = {
        name: gaze_motion_summary(
            cv2,
            name,
            marker_data,
            times,
            valid_corner_rows[name],
            screen_valid,
            float(frame_times[0]),
            float(frame_times[-1]),
            1.0 / video_info.fps,
            args.max_time_delta,
            args.near_distance,
        )
        for name in required
    }
    summary = {
        "scope": "P2 two-clothing-marker torso-local cross-frame experiment",
        "frame_range": {
            "start_frame_inclusive": args.start_frame,
            "end_frame_exclusive": end_frame,
        },
        "parameters": {
            "max_time_delta_seconds": args.max_time_delta,
            "min_marker_area_pixels": args.min_marker_area,
            "near_distance_marker_units": args.near_distance,
        },
        "availability": availability_summary,
        "per_marker_availability": per_marker_availability,
        "cross_frame_landmark_alignment": direction_summaries,
        "marker_local_gaze_motion": gaze_motion,
        "outputs": {
            "samples_csv": str(csv_path),
            "evidence_frames": evidence_frames,
        },
        "interpretation_limits": [
            "The target marker centre is a visible torso landmark, not an independent "
            "manual face/head annotation.",
            "The first frame with both markers available is the reference frame.",
            "Errors combine marker detection noise, shirt deformation, non-coplanarity, "
            "and homography extrapolation.",
            "Same-time reprojection is a separate internal consistency test.",
            "Compensated displacement measures sensitivity to marker motion; it is not "
            "ground-truth gaze accuracy.",
        ],
    }
    summary_path = out_dir / "marker_cross_frame_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

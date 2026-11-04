#!/usr/bin/env python3
"""Validate and descriptively summarise E2 usage-evaluation records.

This module deliberately performs no significance tests and makes no population
inferences. Synthetic fixtures used by tests are not participant data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "E2-A-v1.0"
BUILD_SCOPE = "U1-C1-first-release-three-method-gui"
OUTCOMES = {"success", "partial", "fail", "abandoned", "not_attempted"}
PARTICIPANT_ID_RE = re.compile(r"^E2-P\d{3}$")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def validate_participant_id(value: Any) -> str:
    participant_id = str(value)
    if not PARTICIPANT_ID_RE.fullmatch(participant_id):
        raise ValueError("participant_id must match E2-P###")
    return participant_id


def validate_protocol_build(
    record: Mapping[str, Any],
    expected_protocol: str = PROTOCOL_VERSION,
    expected_build: str = BUILD_SCOPE,
) -> None:
    if record.get("protocol_version") != expected_protocol:
        raise ValueError(
            f"protocol version mismatch: expected {expected_protocol!r}, "
            f"got {record.get('protocol_version')!r}"
        )
    if record.get("build_scope") != expected_build:
        raise ValueError(
            f"build scope mismatch: expected {expected_build!r}, "
            f"got {record.get('build_scope')!r}"
        )


def score_sus(responses: Sequence[Any] | Mapping[Any, Any]) -> float:
    """Return a standard 0--100 SUS score from ten 1--5 responses."""
    if isinstance(responses, Mapping):
        try:
            ordered = [responses[index] for index in range(1, 11)]
        except KeyError:
            try:
                ordered = [responses[str(index)] for index in range(1, 11)]
            except KeyError as exc:
                raise ValueError("SUS requires items 1 through 10") from exc
    else:
        ordered = list(responses)
    if len(ordered) != 10:
        raise ValueError("SUS requires exactly 10 responses")
    contributions: list[float] = []
    for index, raw in enumerate(ordered, start=1):
        response = _finite_number(raw, f"SUS item {index}")
        if response not in {1, 2, 3, 4, 5}:
            raise ValueError(f"SUS item {index} must be an integer from 1 to 5")
        contributions.append(response - 1 if index % 2 else 5 - response)
    return sum(contributions) * 2.5


def _median_range(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "minimum": None, "maximum": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_sus(participants: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    individual: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for record in participants:
        validate_participant_id(record.get("participant_id"))
        validate_protocol_build(record)
        responses = record.get("sus_responses")
        if not isinstance(responses, (Sequence, Mapping)) or isinstance(responses, (str, bytes)):
            missing.append({"participant_id": record["participant_id"], "reason": "responses missing"})
            continue
        if not isinstance(responses, Mapping) and len(responses) != 10:
            missing.append({"participant_id": record["participant_id"], "reason": "not all 10 items completed"})
            continue
        individual.append({
            "participant_id": record["participant_id"],
            "sus_score": score_sus(responses),
        })
    scores = [row["sus_score"] for row in individual]
    summary = _median_range(scores)
    summary["mean_descriptive_only"] = statistics.mean(scores) if scores else None
    return {"individual": individual, "missing": missing, "small_sample_descriptive": summary}


def _split_codes(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, Sequence):
        return [str(part).strip() for part in value if str(part).strip()]
    raise ValueError("coded fields must be semicolon-delimited text or a list")


def summarize_tasks(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    outcomes: Counter[str] = Counter()
    durations_by_task: dict[str, list[float]] = defaultdict(list)
    individual_times: list[dict[str, Any]] = []
    error_types: Counter[str] = Counter()
    error_severity: Counter[str] = Counter()
    totals = Counter()
    for row in rows:
        validate_participant_id(row.get("participant_id"))
        validate_protocol_build(row)
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("task_id is required")
        outcome = str(row.get("outcome", ""))
        if outcome not in OUTCOMES:
            raise ValueError(f"invalid task outcome: {outcome!r}")
        outcomes[outcome] += 1
        duration_raw = row.get("duration_seconds")
        if outcome == "not_attempted" and duration_raw in (None, ""):
            duration = None
        else:
            duration = _finite_number(duration_raw, "duration_seconds")
            if duration < 0:
                raise ValueError("duration_seconds cannot be negative")
        if outcome != "not_attempted":
            assert duration is not None
            durations_by_task[task_id].append(duration)
            individual_times.append({
                "participant_id": row["participant_id"],
                "task_id": task_id,
                "duration_seconds": duration,
            })
        for field in ("error_count", "help_count", "prompt_level"):
            number = _finite_number(row.get(field, 0), field)
            if number < 0 or not number.is_integer():
                raise ValueError(f"{field} must be a non-negative integer")
            totals[field] += int(number)
        error_types.update(_split_codes(row.get("error_types")))
        error_severity.update(_split_codes(row.get("error_severity")))
    return {
        "outcomes": dict(sorted(outcomes.items())),
        "individual_times": individual_times,
        "time_by_task": {
            task_id: _median_range(values)
            for task_id, values in sorted(durations_by_task.items())
        },
        "errors": {
            "total": totals["error_count"],
            "by_type": dict(sorted(error_types.items())),
            "by_severity": dict(sorted(error_severity.items())),
        },
        "help": {"total": totals["help_count"], "prompt_level_sum": totals["prompt_level"]},
    }


def summarize_usefulness(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise each usefulness item separately; never create an overall score."""
    by_item: dict[str, list[float]] = defaultdict(list)
    for row in records:
        validate_participant_id(row.get("participant_id"))
        validate_protocol_build(row)
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            raise ValueError("usefulness item_id is required")
        response = _finite_number(row.get("response"), f"usefulness {item_id}")
        if response not in {1, 2, 3, 4, 5}:
            raise ValueError(f"usefulness {item_id} must be an integer from 1 to 5")
        by_item[item_id].append(response)
    output: dict[str, Any] = {}
    for item_id, values in sorted(by_item.items()):
        distribution = Counter(str(int(value)) for value in values)
        output[item_id] = {
            **_median_range(values),
            "distribution_1_to_5": {str(i): distribution.get(str(i), 0) for i in range(1, 6)},
        }
    return {"items": output, "overall_score": None, "overall_score_validated": False}


def validate_dataset(payload: Mapping[str, Any]) -> None:
    if payload.get("data_class") not in {"consented_anonymized_session_data", "synthetic_unit_test_fixture"}:
        raise ValueError("data_class must identify consented anonymized data or a synthetic unit-test fixture")
    for name in ("participants", "tasks", "usefulness"):
        if not isinstance(payload.get(name, []), list):
            raise ValueError(f"{name} must be a list")
    if payload.get("data_class") == "consented_anonymized_session_data" and not payload.get("ethics_consent_gate_confirmed"):
        raise ValueError("real session data requires a confirmed ethics/consent gate")
    if payload.get("data_class") == "consented_anonymized_session_data":
        for participant in payload.get("participants", []):
            if participant.get("consent_confirmed") is not True:
                raise ValueError("every real participant record requires consent_confirmed=true")


def build_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_dataset(payload)
    return {
        "schema_version": "e2_usage_descriptive_summary.v1",
        "protocol_version": PROTOCOL_VERSION,
        "build_scope": BUILD_SCOPE,
        "data_class": payload["data_class"],
        "inference_policy": "descriptive formative/exploratory evidence only; no population inference",
        "sus": summarize_sus(payload.get("participants", [])),
        "tasks": summarize_tasks(payload.get("tasks", [])),
        "usefulness": summarize_usefulness(payload.get("usefulness", [])),
        "protocol_deviations": list(payload.get("protocol_deviations", [])),
        "missing_data_notes": list(payload.get("missing_data_notes", [])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("input JSON must contain one object")
    summary = build_summary(payload)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output_json), "participant_count": len(payload.get("participants", []))}, indent=2))


if __name__ == "__main__":
    main()

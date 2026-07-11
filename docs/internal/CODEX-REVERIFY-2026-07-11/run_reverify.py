#!/usr/bin/env python3
"""Reasoning-on re-verification with the bounded answer-floor contract."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ANSWER_PASS_FLOOR = 48
PRIOR_HARNESS = (
    Path(__file__).resolve().parents[1]
    / "CODEX-REASONING-STRESS-2026-07-11"
    / "run_api_stress.py"
)

spec = importlib.util.spec_from_file_location("reasoning_stress", PRIOR_HARNESS)
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load prior stress harness: {PRIOR_HARNESS}")
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)

stress.MODES = ("on",)
if requested_routes := os.environ.get("REVERIFY_ROUTES"):
    stress.ROUTES = tuple(
        route.strip() for route in requested_routes.split(",") if route.strip()
    )
_original_grade_turn = stress.grade_turn


def _grade_turn_with_bounded_floor(turn, mode, stream, requested_max_tokens):
    """Allow only the documented <=48-token reasoning answer-pass overage."""
    failures = _original_grade_turn(
        turn,
        mode,
        stream,
        requested_max_tokens,
    )
    usage = turn.get("completion_tokens")
    overage = max(0, int(usage or 0) - requested_max_tokens)
    turn["usage_overage_tokens"] = overage
    turn["allowed_usage_overage_tokens"] = ANSWER_PASS_FLOOR if mode == "on" else 0
    if mode != "on" or usage is None:
        return failures

    failures = [
        failure
        for failure in failures
        if not failure.startswith("completion_usage_over_cap:")
    ]
    if overage > ANSWER_PASS_FLOOR:
        failures.append(
            "completion_usage_over_bounded_floor:"
            f"{usage}>{requested_max_tokens}+{ANSWER_PASS_FLOOR}"
        )
    if stream and turn.get("request", {}).get("model") and turn.get("events"):
        events = turn["events"]
        done_indexes = [
            index
            for index, event in enumerate(events)
            if (event.get("data") or {}).get("done") is True
        ]
        content_indexes = [
            index
            for index, event in enumerate(events)
            if ((event.get("data") or {}).get("message") or {}).get("content")
        ]
        if done_indexes and (
            len(done_indexes) != 1
            or (content_indexes and done_indexes[0] < content_indexes[-1])
        ):
            failures.append(
                "ollama_terminal_before_content_or_not_unique:"
                f"done={done_indexes}:content={content_indexes}"
            )
    return failures


stress.grade_turn = _grade_turn_with_bounded_floor


if __name__ == "__main__":
    raise SystemExit(stress.main())

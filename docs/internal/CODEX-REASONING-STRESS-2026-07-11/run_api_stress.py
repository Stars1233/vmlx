#!/usr/bin/env python3
"""Four-route reasoning/streaming stress against an already-running vMLX server."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROUTES = ("chat", "responses", "anthropic", "ollama")
MODES = ("off", "auto", "on")
MAX_TOKENS = 128
REASONING_MAX_TOKENS = 384
RAW_TAG_RE = re.compile(
    r"</?think>|</?mm:think>|\[/?THINK\]|<tool|</tool|<function|"
    r"hy_User|hy_Assistant",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\w+")
TURN_PROMPTS = (
    "This is turn 1. Remember code ORBIT-731. Respond with three short "
    "sentences; include TURN1 exactly once and ORBIT-731 exactly once.",
    "This is turn 2. State the remembered code and compute 7+5. Respond with "
    "three short sentences; include TURN2 exactly once.",
    "This is turn 3. State both prior turn labels and the remembered code. "
    "Respond with three short sentences and end with FINAL-CHECK.",
)
REASONING_TURN_PROMPTS = (
    "Remember code ORBIT-731 for later. Briefly acknowledge with TURN1 and "
    "the code.",
    "Recall the code and compute 7+5. Briefly answer with TURN2.",
    "Briefly list TURN1, TURN2, and the remembered code, then end with "
    "FINAL-CHECK.",
)


def _post(base: str, path: str, body: dict[str, Any], *, stream: bool):
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=300)


def _controls(route: str, mode: str) -> dict[str, Any]:
    if mode == "auto":
        return {}
    enabled = mode == "on"
    if route == "anthropic":
        if enabled:
            return {"thinking": {"type": "enabled", "budget_tokens": 64}}
        return {"thinking": {"type": "disabled"}}
    if route == "ollama":
        return {"think": enabled}
    return {"enable_thinking": enabled}


def _body(
    route: str,
    model: str,
    mode: str,
    messages: list[dict[str, Any]],
    prompt: str,
    *,
    stream: bool,
    previous_response_id: str | None = None,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, dict[str, Any]]:
    if route == "chat":
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": 731,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        path = "/v1/chat/completions"
    elif route == "responses":
        body = {
            "model": model,
            "input": prompt,
            "temperature": 0,
            "max_output_tokens": max_tokens,
            "seed": 731,
            "stream": stream,
        }
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        path = "/v1/responses"
    elif route == "anthropic":
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": 731,
            "stream": stream,
        }
        path = "/v1/messages"
    elif route == "ollama":
        body = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0,
                "num_predict": max_tokens,
                "seed": 731,
            },
        }
        path = "/api/chat"
    else:  # pragma: no cover
        raise ValueError(route)
    body.update(_controls(route, mode))
    return path, body


def _response_text_and_reasoning(route: str, data: dict[str, Any]):
    if route == "chat":
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return (
            message.get("content") or "",
            message.get("reasoning_content") or message.get("reasoning") or "",
            usage.get("completion_tokens"),
            choice.get("finish_reason"),
            data.get("id"),
        )
    if route == "responses":
        content = data.get("output_text") or ""
        reasoning_parts: list[str] = []
        for item in data.get("output") or []:
            if item.get("type") == "message":
                if not content:
                    content += "".join(
                        part.get("text", "")
                        for part in item.get("content") or []
                        if part.get("type") == "output_text"
                    )
            if item.get("type") == "reasoning":
                reasoning_parts.extend(
                    part.get("text", "")
                    for part in item.get("content") or []
                    if part.get("type") == "reasoning"
                )
        usage = data.get("usage") or {}
        return (
            content,
            "".join(reasoning_parts),
            usage.get("output_tokens"),
            data.get("status"),
            data.get("id"),
        )
    if route == "anthropic":
        texts: list[str] = []
        reasoning: list[str] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                reasoning.append(block.get("thinking", ""))
        return (
            "".join(texts),
            "".join(reasoning),
            (data.get("usage") or {}).get("output_tokens"),
            data.get("stop_reason"),
            data.get("id"),
        )
    message = data.get("message") or {}
    return (
        message.get("content") or "",
        message.get("thinking") or "",
        data.get("eval_count"),
        data.get("done_reason"),
        None,
    )


def _stream_result(route: str, response) -> dict[str, Any]:
    content_deltas: list[str] = []
    reasoning_deltas: list[str] = []
    delta_times: list[float] = []
    events: list[dict[str, Any]] = []
    usage = None
    finish = None
    response_id = None
    done_text = None
    start = time.monotonic()

    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
        else:
            payload = line
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            events.append({"unparsed": line, "t": time.monotonic() - start})
            continue
        event_time = time.monotonic() - start
        events.append({"data": obj, "t": event_time})

        if route == "chat":
            if obj.get("usage"):
                usage = obj["usage"].get("completion_tokens", usage)
            choices = obj.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                finish = choice.get("finish_reason") or finish
                if text:
                    content_deltas.append(text)
                    delta_times.append(event_time)
                if reasoning:
                    reasoning_deltas.append(reasoning)
            response_id = obj.get("id") or response_id
        elif route == "responses":
            kind = obj.get("type")
            if kind == "response.output_text.delta" and obj.get("delta"):
                content_deltas.append(obj["delta"])
                delta_times.append(event_time)
            elif kind == "response.reasoning_summary_text.delta" and obj.get("delta"):
                reasoning_deltas.append(obj["delta"])
            elif kind == "response.output_text.done":
                done_text = obj.get("text")
            elif kind == "response.completed":
                final = obj.get("response") or {}
                response_id = final.get("id") or response_id
                usage = (final.get("usage") or {}).get("output_tokens", usage)
                finish = final.get("status") or finish
        elif route == "anthropic":
            kind = obj.get("type")
            if kind == "message_start":
                response_id = (obj.get("message") or {}).get("id") or response_id
            elif kind == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    content_deltas.append(delta["text"])
                    delta_times.append(event_time)
                elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                    reasoning_deltas.append(delta["thinking"])
            elif kind == "message_delta":
                usage = (obj.get("usage") or {}).get("output_tokens", usage)
                finish = (obj.get("delta") or {}).get("stop_reason") or finish
        else:
            message = obj.get("message") or {}
            text = message.get("content") or ""
            reasoning = message.get("thinking") or ""
            if text:
                content_deltas.append(text)
                delta_times.append(event_time)
            if reasoning:
                reasoning_deltas.append(reasoning)
            if obj.get("done"):
                usage = obj.get("eval_count", usage)
                finish = obj.get("done_reason") or finish

    return {
        "content": "".join(content_deltas),
        "reasoning": "".join(reasoning_deltas),
        "completion_tokens": usage,
        "finish": finish,
        "response_id": response_id,
        "content_deltas": content_deltas,
        "reasoning_deltas": reasoning_deltas,
        "content_delta_times_s": delta_times,
        "done_text": done_text,
        "events": events,
        "elapsed_s": time.monotonic() - start,
    }


def call(
    base: str,
    route: str,
    model: str,
    mode: str,
    messages: list[dict[str, Any]],
    prompt: str,
    *,
    stream: bool,
    previous_response_id: str | None = None,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    path, body = _body(
        route,
        model,
        mode,
        messages,
        prompt,
        stream=stream,
        previous_response_id=previous_response_id,
        max_tokens=max_tokens,
    )
    started = time.monotonic()
    try:
        with _post(base, path, body, stream=stream) as response:
            status = response.status
            if stream:
                result = _stream_result(route, response)
            else:
                raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                content, reasoning, usage, finish, response_id = (
                    _response_text_and_reasoning(route, data)
                )
                result = {
                    "content": content,
                    "reasoning": reasoning,
                    "completion_tokens": usage,
                    "finish": finish,
                    "response_id": response_id,
                    "raw_response": data,
                }
        result.update(
            {
                "ok": status == 200,
                "http_status": status,
                "request": body,
                "elapsed_s": time.monotonic() - started,
            }
        )
        return result
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "http_status": exc.code,
            "error": raw,
            "request": body,
            "elapsed_s": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "request": body,
            "elapsed_s": time.monotonic() - started,
        }


def _loop_score(text: str) -> int:
    words = WORD_RE.findall(text.lower())
    counts: dict[tuple[str, ...], int] = {}
    for i in range(max(0, len(words) - 7)):
        key = tuple(words[i : i + 8])
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def grade_turn(
    turn: dict[str, Any], mode: str, stream: bool, requested_max_tokens: int
) -> list[str]:
    failures: list[str] = []
    if not turn.get("ok"):
        return [f"request_failed:{turn.get('http_status')}:{turn.get('error', '')}"]
    content = turn.get("content") or ""
    reasoning = turn.get("reasoning") or ""
    usage = turn.get("completion_tokens")
    if not content.strip():
        failures.append("empty_content")
    if RAW_TAG_RE.search(content):
        failures.append("raw_tag_in_content")
    if usage is None:
        failures.append("missing_completion_usage")
    elif int(usage) > requested_max_tokens:
        failures.append(
            f"completion_usage_over_cap:{usage}>{requested_max_tokens}"
        )
    if mode == "off" and reasoning.strip():
        failures.append("reasoning_present_when_off")
    if mode == "on" and not reasoning.strip():
        failures.append("reasoning_missing_when_on")
    if _loop_score(content) > 2:
        failures.append(f"repeated_8gram:{_loop_score(content)}")
    if stream:
        deltas = turn.get("content_deltas") or []
        if len(deltas) < 2:
            failures.append(f"content_not_incremental:{len(deltas)}_delta")
        times = turn.get("content_delta_times_s") or []
        if len(times) >= 2 and times[-1] <= times[0]:
            failures.append("content_delta_timestamps_not_increasing")
        if turn.get("done_text") is not None and turn.get("done_text") != content:
            failures.append("stream_done_text_mismatch")
    return failures


def run_sequence(base: str, route: str, model: str, mode: str, stream: bool):
    messages: list[dict[str, Any]] = []
    previous_response_id = None
    turns: list[dict[str, Any]] = []
    requested_max_tokens = REASONING_MAX_TOKENS if mode == "on" else MAX_TOKENS
    prompts = REASONING_TURN_PROMPTS if mode == "on" else TURN_PROMPTS
    for index, prompt in enumerate(prompts, start=1):
        if route != "responses":
            messages.append({"role": "user", "content": prompt})
        turn = call(
            base,
            route,
            model,
            mode,
            messages,
            prompt,
            stream=stream,
            previous_response_id=previous_response_id,
            max_tokens=requested_max_tokens,
        )
        turn["turn"] = index
        turn["requested_max_tokens"] = requested_max_tokens
        turn["failures"] = grade_turn(
            turn, mode, stream, requested_max_tokens
        )
        turns.append(turn)
        if route == "responses":
            previous_response_id = turn.get("response_id")
        else:
            messages.append(
                {"role": "assistant", "content": turn.get("content") or ""}
            )
        print(
            f"{route:10s} {mode:4s} {'stream' if stream else 'nonstream':9s} "
            f"turn={index} usage={turn.get('completion_tokens')} "
            f"content_deltas={len(turn.get('content_deltas') or [])} "
            f"failures={turn['failures']}",
            flush=True,
        )

    combined = "\n".join(turn.get("content") or "" for turn in turns)
    semantic_failures: list[str] = []
    if "ORBIT-731" not in combined:
        semantic_failures.append("multiturn_code_recall_missing")
    if "TURN1" not in combined or "TURN2" not in combined:
        semantic_failures.append("multiturn_label_recall_missing")
    if "FINAL-CHECK" not in (turns[-1].get("content") or ""):
        semantic_failures.append("final_marker_missing")
    return {
        "route": route,
        "mode": mode,
        "stream": stream,
        "requested_max_tokens": requested_max_tokens,
        "turns": turns,
        "semantic_failures": semantic_failures,
        "failures": [
            f"turn{turn['turn']}:{failure}"
            for turn in turns
            for failure in turn["failures"]
        ]
        + semantic_failures,
    }


def determinism_probe(base: str, route: str, model: str):
    prompt = "Reply exactly DET-731."
    messages = [{"role": "user", "content": prompt}]
    runs = [
        call(
            base,
            route,
            model,
            "off",
            messages,
            prompt,
            stream=False,
            max_tokens=48,
        )
        for _ in range(3)
    ]
    measured = runs[1:]
    byte_identical = all(
        run.get("content") == measured[0].get("content")
        and run.get("reasoning") == measured[0].get("reasoning")
        for run in measured[1:]
    )
    failures = [] if byte_identical else ["warm_greedy_not_byte_identical"]
    for index, run in enumerate(measured, start=1):
        usage = run.get("completion_tokens")
        if usage is None or int(usage) > 48:
            failures.append(f"run{index}_usage_invalid:{usage}")
    print(
        f"{route:10s} determinism warm_equal={byte_identical} "
        f"usages={[run.get('completion_tokens') for run in measured]}",
        flush=True,
    )
    return {
        "route": route,
        "warmup": runs[0],
        "measured": measured,
        "byte_identical": byte_identical,
        "failures": failures,
    }


def regrade_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute grades from captured wire evidence without issuing requests."""
    for row in report.get("sequences") or []:
        route = row["route"]
        mode = row["mode"]
        stream = bool(row["stream"])
        requested_max_tokens = int(
            row.get("requested_max_tokens")
            or (REASONING_MAX_TOKENS if mode == "on" else MAX_TOKENS)
        )
        for turn in row.get("turns") or []:
            if not stream and isinstance(turn.get("raw_response"), dict):
                content, reasoning, usage, finish, response_id = (
                    _response_text_and_reasoning(route, turn["raw_response"])
                )
                turn.update(
                    {
                        "content": content,
                        "reasoning": reasoning,
                        "completion_tokens": usage,
                        "finish": finish,
                        "response_id": response_id,
                    }
                )
            turn["failures"] = grade_turn(
                turn, mode, stream, requested_max_tokens
            )
        turns = row.get("turns") or []
        combined = "\n".join(turn.get("content") or "" for turn in turns)
        semantic_failures: list[str] = []
        if "ORBIT-731" not in combined:
            semantic_failures.append("multiturn_code_recall_missing")
        if "TURN1" not in combined or "TURN2" not in combined:
            semantic_failures.append("multiturn_label_recall_missing")
        if turns and "FINAL-CHECK" not in (turns[-1].get("content") or ""):
            semantic_failures.append("final_marker_missing")
        row["semantic_failures"] = semantic_failures
        row["failures"] = [
            f"turn{turn['turn']}:{failure}"
            for turn in turns
            for failure in turn["failures"]
        ] + semantic_failures

    failures = [
        f"{row['route']}/{row['mode']}/{'stream' if row['stream'] else 'nonstream'}:{failure}"
        for row in report.get("sequences") or []
        for failure in row["failures"]
    ]
    failures.extend(
        f"{row['route']}/determinism:{failure}"
        for row in report.get("determinism") or []
        for failure in row.get("failures") or []
    )
    report["failures"] = failures
    report["status"] = "pass" if not failures else "fail"
    return report


def write_report_outputs(path: Path, report: dict[str, Any]) -> Path:
    """Write the full report and a compact bundle containing every failed row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    failures_path = path.with_name(f"{path.stem}-failures{path.suffix}")
    failed_sequences = []
    for row in report.get("sequences") or []:
        if not row.get("failures"):
            continue
        failed_sequences.append(
            {
                key: row.get(key)
                for key in (
                    "route",
                    "mode",
                    "stream",
                    "requested_max_tokens",
                    "semantic_failures",
                    "failures",
                )
            }
            | {
                "turns": [
                    {
                        key: turn.get(key)
                        for key in (
                            "turn",
                            "request",
                            "http_status",
                            "content",
                            "reasoning",
                            "completion_tokens",
                            "finish",
                            "response_id",
                            "content_deltas",
                            "reasoning_deltas",
                            "content_delta_times_s",
                            "done_text",
                            "elapsed_s",
                            "failures",
                        )
                    }
                    for turn in row.get("turns") or []
                ]
            }
        )
    failures_path.write_text(
        json.dumps(
            {
                "base": report.get("base"),
                "model": report.get("model"),
                "max_tokens": report.get("max_tokens"),
                "failed_sequences": failed_sequences,
                "failed_determinism": [
                    row
                    for row in report.get("determinism") or []
                    if row.get("failures")
                ],
                "failures": report.get("failures") or [],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return failures_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="jangq-ai/Hy3-JANG_2K-MTP")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regrade-existing", action="store_true")
    args = parser.parse_args()

    if args.regrade_existing:
        report = regrade_report(json.loads(args.out.read_text()))
        failures_path = write_report_outputs(args.out, report)
        print(
            f"FINAL status={report['status']} failures={len(report['failures'])} "
            f"out={args.out} failures_out={failures_path}"
        )
        return 0 if not report["failures"] else 1

    report: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base": args.base,
        "model": args.model,
        "max_tokens": {
            "off_auto": MAX_TOKENS,
            "on": REASONING_MAX_TOKENS,
        },
        "sequences": [],
        "determinism": [],
    }
    for route in ROUTES:
        for mode in MODES:
            for stream in (False, True):
                report["sequences"].append(
                    run_sequence(args.base, route, args.model, mode, stream)
                )
    for route in ROUTES:
        report["determinism"].append(
            determinism_probe(args.base, route, args.model)
        )

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    regrade_report(report)
    failures_path = write_report_outputs(args.out, report)
    print(
        f"FINAL status={report['status']} failures={len(report['failures'])} "
        f"out={args.out} failures_out={failures_path}"
    )
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from typing import Any

import requests


BASE = "http://127.0.0.1:8088"
MODEL = "JANGQ-AI/MiniMax-M3-Coder-Small"
RESULT = (
    "Path: panel/package.json\nType: file\nSize: 5.2 KB\n"
    "Modified: 2026-07-15T14:28:24.904Z\nPermissions: 0644"
)

RESPONSES_TOOL = {
    "type": "function",
    "name": "file_info",
    "description": "Return current filesystem metadata for a path.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "file_info",
        "description": "Return current filesystem metadata for a path.",
        "parameters": RESPONSES_TOOL["parameters"],
    },
}


def post_stream(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with requests.post(BASE + path, json=payload, stream=True, timeout=240) as response:
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if not raw:
                continue
            row: dict[str, Any] = {"t": round(time.monotonic() - started, 6), "raw": raw}
            if raw.startswith("data: "):
                value = raw[6:]
                if value == "[DONE]":
                    row["data"] = "[DONE]"
                else:
                    try:
                        row["data"] = json.loads(value)
                    except json.JSONDecodeError:
                        row["parse_error"] = value
            rows.append(row)
    return {"elapsed": round(time.monotonic() - started, 6), "events": rows}


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    response = requests.post(BASE + path, json=payload, timeout=240)
    elapsed = round(time.monotonic() - started, 6)
    body: Any
    try:
        body = response.json()
    except Exception:
        body = response.text
    return {"status_code": response.status_code, "elapsed": elapsed, "body": body}


def summarize_responses(row: dict[str, Any]) -> dict[str, Any]:
    reasoning: list[str] = []
    content: list[str] = []
    functions: list[dict[str, Any]] = []
    terminals: list[str] = []
    counts: dict[str, int] = {}
    for event in row["events"]:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        kind = str(data.get("type") or "")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "response.reasoning_summary_text.delta":
            reasoning.append(str(data.get("delta") or ""))
        elif kind == "response.output_text.delta":
            content.append(str(data.get("delta") or ""))
        elif kind == "response.output_item.done":
            item = data.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                functions.append(item)
        elif kind in {"response.completed", "response.incomplete", "response.failed"}:
            terminals.append(kind)
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "functions": functions,
        "event_counts": counts,
        "terminals": terminals,
    }


def summarize_chat(row: dict[str, Any]) -> dict[str, Any]:
    reasoning: list[str] = []
    content: list[str] = []
    tool_parts: dict[int, dict[str, Any]] = {}
    finish: list[str] = []
    done = 0
    for event in row["events"]:
        data = event.get("data")
        if data == "[DONE]":
            done += 1
            continue
        if not isinstance(data, dict):
            continue
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning.append(str(delta.get("reasoning_content") or ""))
            content.append(str(delta.get("content") or ""))
            for part in delta.get("tool_calls") or []:
                index = int(part.get("index") or 0)
                target = tool_parts.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if part.get("id"):
                    target["id"] = part["id"]
                fn = part.get("function") or {}
                target["function"]["name"] += str(fn.get("name") or "")
                target["function"]["arguments"] += str(fn.get("arguments") or "")
            if choice.get("finish_reason") is not None:
                finish.append(str(choice["finish_reason"]))
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": [tool_parts[k] for k in sorted(tool_parts)],
        "finish_reasons": finish,
        "done_count": done,
    }


def response_call(summary: dict[str, Any]) -> tuple[str, str, str]:
    call = summary["functions"][0]
    return str(call.get("call_id") or call.get("id")), str(call.get("name")), str(call.get("arguments"))


def chat_call(summary: dict[str, Any]) -> tuple[str, str, str]:
    call = summary["tool_calls"][0]
    return str(call.get("id")), str(call["function"].get("name")), str(call["function"].get("arguments"))


def main() -> None:
    output_path = sys.argv[1]
    no_tool_prompt = (
        "Do not call a tool. In two concise sentences, explain why progressive "
        "answer deltas matter after a private reasoning phase."
    )
    tool_prompt = (
        "Call file_info exactly once with path panel/package.json. Do not answer "
        "before emitting the tool call."
    )
    follow_prompt = (
        "Using only that current tool result, do not call another tool. Reply "
        "exactly M3-GATEWAY-TOOL-DONE SIZE=5.2 KB and nothing else."
    )
    common = {
        "model": MODEL,
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 768,
        "enable_thinking": True,
    }
    out: dict[str, Any] = {"model": MODEL, "responses": {}, "chat": {}}

    r_no_payload = {
        **common,
        "input": no_tool_prompt,
        "tools": [RESPONSES_TOOL],
        "tool_choice": "none",
        "stream": True,
    }
    r_no = post_stream("/v1/responses", r_no_payload)
    r_no["summary"] = summarize_responses(r_no)
    out["responses"]["no_tool_stream"] = r_no

    r_tool_payload = {
        **common,
        "input": tool_prompt,
        "tools": [RESPONSES_TOOL],
        "tool_choice": "required",
        "stream": True,
    }
    r_tool = post_stream("/v1/responses", r_tool_payload)
    r_tool["summary"] = summarize_responses(r_tool)
    out["responses"]["tool_stream"] = r_tool
    r_call_id, r_name, r_args = response_call(r_tool["summary"])

    r_follow_input = [
        {"role": "user", "content": tool_prompt},
        {"type": "function_call", "call_id": r_call_id, "name": r_name, "arguments": r_args},
        {"type": "function_call_output", "call_id": r_call_id, "output": RESULT},
        {"role": "user", "content": follow_prompt},
    ]
    r_follow_payload = {
        **common,
        "input": r_follow_input,
        "tools": [RESPONSES_TOOL],
        "tool_choice": "none",
        "stream": True,
    }
    r_follow = post_stream("/v1/responses", r_follow_payload)
    r_follow["summary"] = summarize_responses(r_follow)
    out["responses"]["follow_stream"] = r_follow
    out["responses"]["no_tool_nonstream"] = post_json(
        "/v1/responses", {**r_no_payload, "stream": False}
    )
    out["responses"]["follow_nonstream"] = post_json(
        "/v1/responses", {**r_follow_payload, "stream": False}
    )

    chat_common = {
        "model": MODEL,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 768,
        "enable_thinking": True,
    }
    c_no_payload = {
        **chat_common,
        "messages": [{"role": "user", "content": no_tool_prompt}],
        "tools": [CHAT_TOOL],
        "tool_choice": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    c_no = post_stream("/v1/chat/completions", c_no_payload)
    c_no["summary"] = summarize_chat(c_no)
    out["chat"]["no_tool_stream"] = c_no

    c_tool_payload = {
        **chat_common,
        "messages": [{"role": "user", "content": tool_prompt}],
        "tools": [CHAT_TOOL],
        "tool_choice": "required",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    c_tool = post_stream("/v1/chat/completions", c_tool_payload)
    c_tool["summary"] = summarize_chat(c_tool)
    out["chat"]["tool_stream"] = c_tool
    c_call_id, c_name, c_args = chat_call(c_tool["summary"])
    c_messages = [
        {"role": "user", "content": tool_prompt},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": c_call_id,
                    "type": "function",
                    "function": {"name": c_name, "arguments": c_args},
                }
            ],
        },
        {"role": "tool", "tool_call_id": c_call_id, "content": RESULT},
        {"role": "user", "content": follow_prompt},
    ]
    c_follow_payload = {
        **chat_common,
        "messages": c_messages,
        "tools": [CHAT_TOOL],
        "tool_choice": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    c_follow = post_stream("/v1/chat/completions", c_follow_payload)
    c_follow["summary"] = summarize_chat(c_follow)
    out["chat"]["follow_stream"] = c_follow
    out["chat"]["no_tool_nonstream"] = post_json(
        "/v1/chat/completions", {**c_no_payload, "stream": False}
    )
    out["chat"]["follow_nonstream"] = post_json(
        "/v1/chat/completions", {**c_follow_payload, "stream": False}
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")

    compact = {
        ep: {
            name: value.get("summary", value)
            for name, value in rows.items()
        }
        for ep, rows in out.items()
        if ep in {"responses", "chat"}
    }
    print(json.dumps(compact, indent=2)[:30000])


if __name__ == "__main__":
    main()

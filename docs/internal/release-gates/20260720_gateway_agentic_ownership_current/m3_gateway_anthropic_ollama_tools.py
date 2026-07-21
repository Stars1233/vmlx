#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8088"
MODEL = "JANGQ-AI/MiniMax-M3-Coder-Small"
RESULT = (
    "Path: panel/package.json\nType: file\nSize: 5.2 KB\n"
    "Modified: 2026-07-20T03:54:32.142Z\nPermissions: 0644"
)
TOOL_PROMPT = (
    "Call file_info exactly once with path panel/package.json. Do not answer "
    "before the tool call."
)
FOLLOW_PROMPT = (
    "Using only that tool result, do not call another tool. Reply exactly "
    "M3-GATEWAY-TOOL-DONE SIZE=5.2 KB and nothing else."
)
PARAMETERS = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}
ANTHROPIC_TOOL = {
    "name": "file_info",
    "description": "Return current filesystem metadata for a path.",
    "input_schema": PARAMETERS,
}
OLLAMA_TOOL = {
    "type": "function",
    "function": {
        "name": "file_info",
        "description": "Return current filesystem metadata for a path.",
        "parameters": PARAMETERS,
    },
}


def anthropic_stream(payload: dict) -> dict:
    started = time.monotonic()
    events = []
    event_name = None
    with requests.post(
        BASE + "/v1/messages", json=payload, stream=True, timeout=240
    ) as response:
        status = response.status_code
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                event_name = None
                continue
            if line.startswith("event: "):
                event_name = line[7:]
                continue
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            events.append(
                {"at": round(time.monotonic() - started, 6), "event": event_name, "data": data}
            )

    reasoning = []
    content = []
    tool_id = ""
    tool_name = ""
    tool_json = []
    terminals = []
    for row in events:
        data = row["data"]
        kind = data.get("type") or row.get("event")
        delta = data.get("delta") or {}
        if kind == "content_block_delta" and delta.get("type") in {
            "thinking_delta",
            "reasoning_delta",
        }:
            reasoning.append(delta.get("thinking") or delta.get("reasoning") or "")
        if kind == "content_block_delta" and delta.get("type") == "text_delta":
            content.append(delta.get("text") or "")
        if kind == "content_block_delta" and delta.get("type") == "input_json_delta":
            tool_json.append(delta.get("partial_json") or "")
        if kind == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_id = block.get("id") or ""
                tool_name = block.get("name") or ""
                if block.get("input"):
                    tool_json.append(json.dumps(block["input"]))
        if kind in {"message_stop", "error"}:
            terminals.append(kind)
    return {
        "status": status,
        "elapsed": round(time.monotonic() - started, 6),
        "events": events,
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "reasoning_deltas": len([x for x in reasoning if x]),
        "content_deltas": len([x for x in content if x]),
        "tool_id": tool_id,
        "tool_name": tool_name,
        "tool_arguments": "".join(tool_json),
        "terminals": terminals,
    }


def ollama_stream(payload: dict) -> dict:
    started = time.monotonic()
    rows = []
    with requests.post(
        BASE + "/api/chat", json=payload, stream=True, timeout=240
    ) as response:
        status = response.status_code
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if not raw:
                continue
            rows.append(
                {"at": round(time.monotonic() - started, 6), "data": json.loads(raw)}
            )

    reasoning = []
    content = []
    tools = []
    terminals = []
    for row in rows:
        data = row["data"]
        message = data.get("message") or {}
        if message.get("thinking") or message.get("reasoning"):
            reasoning.append(message.get("thinking") or message.get("reasoning"))
        if message.get("content"):
            content.append(message["content"])
        tools.extend(message.get("tool_calls") or [])
        if data.get("done"):
            terminals.append(data.get("done_reason") or "done")
    return {
        "status": status,
        "elapsed": round(time.monotonic() - started, 6),
        "events": rows,
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "reasoning_deltas": len([x for x in reasoning if x]),
        "content_deltas": len([x for x in content if x]),
        "tool_calls": tools,
        "terminals": terminals,
    }


anthropic_initial = anthropic_stream(
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": TOOL_PROMPT}],
        "tools": [ANTHROPIC_TOOL],
        "tool_choice": {"type": "any"},
        "stream": True,
        "max_tokens": 768,
        "temperature": 0,
        "enable_thinking": True,
    }
)
anthropic_args = json.loads(anthropic_initial["tool_arguments"] or "{}")
anthropic_follow = anthropic_stream(
    {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": TOOL_PROMPT},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": anthropic_initial["tool_id"],
                        "name": anthropic_initial["tool_name"],
                        "input": anthropic_args,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": anthropic_initial["tool_id"],
                        "content": RESULT,
                    },
                    {"type": "text", "text": FOLLOW_PROMPT},
                ],
            },
        ],
        "tools": [ANTHROPIC_TOOL],
        "tool_choice": {"type": "none"},
        "stream": True,
        "max_tokens": 768,
        "temperature": 0,
        "enable_thinking": True,
    }
)

ollama_initial = ollama_stream(
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": TOOL_PROMPT}],
        "tools": [OLLAMA_TOOL],
        "stream": True,
        "think": True,
        "options": {"temperature": 0, "num_predict": 768},
    }
)
ollama_calls = ollama_initial["tool_calls"]
ollama_follow = ollama_stream(
    {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": TOOL_PROMPT},
            {"role": "assistant", "content": "", "tool_calls": ollama_calls},
            {"role": "tool", "tool_name": "file_info", "content": RESULT},
            {"role": "user", "content": FOLLOW_PROMPT},
        ],
        "tools": [OLLAMA_TOOL],
        "stream": True,
        "think": True,
        "options": {"temperature": 0, "num_predict": 768},
    }
)

out = {
    "model": MODEL,
    "anthropic": {"initial": anthropic_initial, "follow": anthropic_follow},
    "ollama": {"initial": ollama_initial, "follow": ollama_follow},
}
out["checks"] = {
    "anthropic_initial_exact_tool": (
        anthropic_initial["status"] == 200
        and anthropic_initial["tool_name"] == "file_info"
        and anthropic_args == {"path": "panel/package.json"}
        and not anthropic_initial["content"]
    ),
    "anthropic_follow_exact_progressive_terminal": (
        anthropic_follow["status"] == 200
        and anthropic_follow["content"] == "M3-GATEWAY-TOOL-DONE SIZE=5.2 KB"
        and anthropic_follow["content_deltas"] > 1
        and anthropic_follow["terminals"] == ["message_stop"]
        and not anthropic_follow["tool_name"]
    ),
    "ollama_initial_exact_tool": (
        ollama_initial["status"] == 200
        and len(ollama_calls) == 1
        and (ollama_calls[0].get("function") or {}).get("name") == "file_info"
        and (ollama_calls[0].get("function") or {}).get("arguments")
        == {"path": "panel/package.json"}
        and not ollama_initial["content"]
    ),
    "ollama_follow_exact_progressive_terminal": (
        ollama_follow["status"] == 200
        and ollama_follow["content"] == "M3-GATEWAY-TOOL-DONE SIZE=5.2 KB"
        and ollama_follow["content_deltas"] > 1
        and ollama_follow["terminals"] == ["stop"]
        and not ollama_follow["tool_calls"]
    ),
}
Path("/tmp/m3-gateway-anthropic-ollama-tools.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False) + "\n"
)
print(
    json.dumps(
        {
            "checks": out["checks"],
            "anthropic": {
                "initial": {k: anthropic_initial[k] for k in ("status", "reasoning_deltas", "content_deltas", "tool_id", "tool_name", "tool_arguments", "terminals")},
                "follow": {k: anthropic_follow[k] for k in ("status", "reasoning_deltas", "content_deltas", "content", "tool_name", "terminals")},
            },
            "ollama": {
                "initial": {k: ollama_initial[k] for k in ("status", "reasoning_deltas", "content_deltas", "tool_calls", "terminals")},
                "follow": {k: ollama_follow[k] for k in ("status", "reasoning_deltas", "content_deltas", "content", "tool_calls", "terminals")},
            },
        },
        indent=2,
        ensure_ascii=False,
    )
)

# Bonsai exact-once streaming and gateway proof — 2026-07-16

Verdict for this scoped row: PASS on current source. This is not a release-wide
PASS; the remaining model/cache/media matrix is still open.

## Source trace

- `vmlx_engine/server.py::stream_responses_api` detects only explicit requests
  that name one exposed tool and require it exactly once.
- Those requests buffer the visible content rail from token one, continue to
  emit genuine `response.reasoning_summary_text.delta` events, and discard
  premature post-`</think>` meta prose once a schema-valid call is finalized.
- General multi-tool/interleaved turns retain the prior streaming behavior.
- `tests/test_server.py` covers both post-call repetition and premature
  post-think visible meta-reasoning.

## Current live evidence

- Gateway Responses, current source, 1-bit Bonsai:
  - first pass: 132 output tokens, reasoning-summary deltas present,
    zero `response.output_text.delta`, empty `response.output_text.done`, exactly
    one `file_info({"path":"panel/package.json"})`.
  - function-output pass: nine output tokens, streamed and finalized exactly as
    `GW-B1-FIX1-DONE`.
- Electron 1-bit Bonsai, one chat across a process restart:
  - DB rows 1839, 1842, 1845, 1848: 96/167/94/98 output tokens.
  - every row has exactly one tool call, one tool result, and exact final text.
- Electron ternary Bonsai, one chat across a process restart:
  - rows 1827, 1830, 1833, 1836 all completed one tool and exact final text.
  - row 1827 was a 1,201-token sampling outlier; later turns were 78/87/86.
  - direct repeated first-pass probes were 84–122 tokens. The parser/API leak
    exposed by the outlier is fixed above; native sampling length remains
    bundle-controlled (`temperature=1.0`, `top_p=0.95`, `top_k=20`).
- Gateway/UI parity:
  - reasoning UI: Auto selected; server argv: `--reasoning-parser qwen3`.
  - cache UI: paged and block L2 checked; 64-token blocks; 1,000 blocks;
    Auto cache quantization selected.
  - 1-bit argv: paged cache and block L2 flags present, no forced cache codec.
  - single-model mode switched ternary to 1-bit: ternary stopped, only 1-bit
    PID 74596 remained.
  - LAN toggle bound dev Electron to `*:8081`; localhost toggle rebound it to
    `127.0.0.1:8081`. Port 8080 was already owned by the separate installed
    vMLX process and was intentionally not killed.

## Tests

- `tests/test_server.py`: 102 passed, 3 deselected.
- Focused exact-once/minimax/output-index selection: 4 passed.
- `py_compile vmlx_engine/server.py`: passed.
- `git diff --check -- vmlx_engine/server.py tests/test_server.py`: passed.

## Still open

- Qwen hybrid SSM companion disk restore reports `restore_enabled=false`; safe
  restart full-prefill is current behavior. Cross-restart SSM codec parity is
  not claimed here.
- DSV4, Pangu, Laguna, HY3 MTP, MiniMax, Mistral/Step, media, and remaining
  cross-protocol rows still require current source plus live Electron evidence.
- Packaging, signing, notarization, tag/feed publication, and release readiness
  remain locked.

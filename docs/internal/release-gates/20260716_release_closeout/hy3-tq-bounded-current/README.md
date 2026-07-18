# HY3 current-head bounded TQ4 and agent-stream proof

Status: `SCOPED_PASS`; global release status: `PARTIAL_NO_RELEASE`.

Date: 2026-07-17 PT
Source commit: `45c64f85e8b4`
Branch: `reconcile/1.5.68`
Push target: `origin/codex/live-electron-gates-20260715`

## Retained failure controls

- `/Users/eric/Library/Logs/DiagnosticReports/python3.13-2026-07-17-165653.ips`
  is the pre-fix HY3 PID 77153 crash. It records `SIGABRT` on
  `com.Metal.CompletionQueueDispatch`; the kernel separately reported 400,000
  leaked IOGPU resources.
- The first direct-storage-encoder attempt still accumulated all lazy
  page/layer graphs before writing. A new 7,861-token cold request completed
  model output but UI logs reached `[metal::malloc] Resource limit (499000)`
  across the later layers and health stayed at zero disk writes.
- These failures are controls, not passes. They established that eliminating
  live-cache decode/join allocation was necessary but not sufficient.

## Source trace

- `vmlx_engine/tq_disk_store.py::_tq_decoder_pair` constructs only the two
  immutable `TurboQuantEncoder` objects using the same key seed and value
  `seed + 500` policy as `TurboQuantKVCache._ensure_encoders`.
- `vmlx_engine/tq_disk_store.py::encode_tq_block` calls `encode_keys` and
  `encode_values` directly. It no longer instantiates a live cache, calls
  `compress()`, decodes packed values, or creates joined live-attention
  buffers.
- `vmlx_engine/prefix_cache.py` detects a native-TQ disk payload and invokes
  `write_block_async` for that complete page immediately. The disk store
  evaluates and saves that page on the inference thread, while its background
  thread still owns only rename/index work.
- `tests/test_tq_paged_block_cache.py` fails if disk encoding invokes live
  `TurboQuantKVCache.compress()` and pins the bounded
  `extract, write, extract, write` order across two pages.
- Exact numbered excerpts are preserved in `source-tq-disk.txt`,
  `source-prefix-write.txt`, and `source-tests.txt`.

## Focused verification

- `test-tq-paged.txt`: 15/15 passed.
- `test-adjacent-cache.txt`: 35/35 passed across
  `test_turboquant_cache_contract.py`, `test_prefix_cache.py`, and
  `test_terminal_dispatch_before_cache_cleanup.py`.
- The source commit and pushed branch head are recorded in `commit.txt`.

## Matched long-prefix live proof

Artifact: `/Volumes/EricsLLMDrive/jangq-ai/Hy3-JANG_2K-MTP`
Port: `8010`
Policy: q4 native-TQ stored full KV, paged RAM/L2, native MTP depth 1.

The prompt and sampling inputs were byte-identical across the three D rows.
Each response emitted ten `response.output_text.delta` events, returned exact
`HY3-TQ-TTFT-D=583`, and emitted one `response.completed` terminal.

| Route | First content | Cache evidence |
|---|---:|---|
| Cold | 23.073s | 9,061 cache-key tokens stored as 142 native-TQ pages |
| Same process | 5.802s | 9,061 `paged+tq-native` tokens; 3.98x faster than cold |
| Electron Stop/Start, L2 retained | 10.763s | 9,061 `paged+disk+tq-native`; 142 disk and 142 native-TQ hits; 2.14x faster than cold |

The Electron Logs surface counted 142 `writing bounded TQ block` entries,
zero `Resource limit` entries, and zero `TurboQuant compress telemetry`
entries for the bounded cold/warm process. `crash-reports-after.txt` contains
only the retained pre-fix crash; no second `.ips` appeared.

Primary artifacts: `hy3-d-cold-bounded.out`, `hy3-d-warm-bounded.out`,
`hy3-d-disk-bounded.out`, `hy3-d-disk-health.json`,
`hy3-d-bounded-warm-logs.png`, and `electron-log-summary.txt`.

## Responses and Electron agent streaming

- Electron row 369, Auto thinking: 759 reasoning stream events, then 76
  visible content stream events from 27.455s through 30.077s, then completion.
  The persisted/final answer is six coherent lines ending exactly
  `HY3-UI-STREAM1-DONE`.
- Same-chat row 372 at the existing stochastic temperature 0.90 executed one
  matching `file_info(panel/package.json)` call/result but emitted an
  unnumbered draft and a correction before its final six lines. This is kept
  as a strict-format/model-reliability miss in `hy3-ui-tool2-db.json` and
  `hy3-tool2-final.png`.
- A raw deterministic Responses continuation emitted 704 reasoning and 60
  content deltas cold. The byte-identical repeat used 457/460
  `paged+tq-native` input tokens, emitted 169 reasoning and 57 content deltas,
  preserved the exact timestamp, and completed once. This is preserved in
  `hy3-responses-tool-cont*.out`.
- Same-chat Electron row 375 at explicit temperature 0 executed exactly one
  `file_info(vmlx_engine/tq_disk_store.py)`. The DOM observer recorded the
  answer growing from 1 to 135 characters while generation remained active.
  SQLite persists one OpenAI function call, one matching result, exact six
  numbered lines, no warning, and 5,629 `paged+tq-native` cached tokens.

The deterministic rows establish that neither the shared Responses streamer
nor q4 prefix reuse deterministically corrupts this continuation. The retained
0.90 row is not rewritten, deduplicated, or hidden by application code.

## Boundary

This evidence closes the current HY3 resource-lifetime fix, matched q4
cold/RAM/disk reuse, native MTP-D1 activity, Responses reasoning/content
streaming, and same-chat automatic tool continuation. It does not clear other
model families, media paths, remaining settings/protocol rows, the full suite,
packaging, signing, notarization, feeds, PyPI, GitHub release, or public release
readiness.

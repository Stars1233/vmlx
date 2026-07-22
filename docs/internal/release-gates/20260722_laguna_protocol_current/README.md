# Laguna S2.1 / reasoning rail / q4 cache current checkpoint — 2026-07-21

Status: **PARTIAL_NO_RELEASE**

This directory records the current post-v1.6.14 Laguna S2.1 work on
`erics-m5-max.local:/Users/eric/mlx/vllm-mlx-release-1.6.13`.

Do not treat this as a release-ready matrix closure. It is a scoped checkpoint
for:

- Laguna S2.1 UI/API rail separation;
- content-identical reasoning persistence guard;
- q4 TurboQuant cache telemetry;
- one-model startup pruning/adoption;
- retained negative rows that still need broader follow-up.

## Current source trace

- `3431c86b3 fix(chat): drop content-identical reasoning rails`
  - `panel/src/main/ipc/chat.ts`
  - `panel/tests/reasoning-display.test.ts`
- `2ad94974f fix(sessions): prune orphan engines in single-model mode`
- `b6d12fa62 fix(panel): sweep unmanaged engines in single-model mode`
- `ee227bcdc fix(lfm): bind natural file tool paths`
- `1c8ef8ecd fix(cache): invalidate stale mixed-swa l2 blocks`
- `e16a3dac1 fix(api): resolve Laguna auto thinking for responses`
- `2648d7bfd fix(streaming): seed reasoning from registry templates`
- `b20786043 fix(cache): separate generation prompt prefix states`

## Focused tests run in this turn

```text
panel: npm test -- --run tests/reasoning-display.test.ts tests/session-single-model-start.test.ts
result: 2 files passed, 116 tests passed

panel: npm run typecheck
result: passed

python: pytest -q tests/test_tool_prompt_fallback.py -k "natural_file_info or direct_file_info_binds_explicit_path_without_placeholder"
result: 3 passed, 22 deselected
```

Full Python/panel suites were not rerun in this turn.

## Live Electron evidence

Patched dev Electron was relaunched with:

- profile: `/Users/eric/.vmlx-v1613-responsive-dev`
- CDP: `127.0.0.1:9335`
- source checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13/panel`

Evidence file:

- `laguna-railguard-electron-log-3431c86b3.txt`

Key lines captured there:

- `DevTools listening on ws://127.0.0.1:9335/...`
- `[SESSIONS] single-model mode: pruning detected engine pid=64327 port=8051 model=/Volumes/EricsLLMDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK during adoption; keeping pid=62411 port=8008`
- `[STARTUP] Adopted 1 vmlx-engine process(es):`
- `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`
- `[CHAT] Stream ended — content: 19 chars, reasoning: 0 chars, tool calls: 0, buffered: false`
- `[CHAT] Response complete: 10 tokens in 0.8s ... pp: 593 tokens (524 cached) ... usage=server`

The live process list after startup contained only the Laguna engine:

```text
/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine serve /Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L --port 8008 ...
```

The stray LFM engine on `8051` was present before patched startup and absent
after patched startup.

## Live UI / DB rail evidence

Evidence file:

- `laguna-railguard-live1-db-3431c86b3.json`

Prompt:

```text
[LAG-S21-RAILGUARD-LIVE1] Do not call tools. Think briefly if reasoning is enabled, but keep all reasoning out of the visible answer. The visible answer must be exactly: LAGUNA RAILGUARD OK
```

Persisted assistant row:

```json
{
  "content": "LAGUNA RAILGUARD OK",
  "reasoning_content": null,
  "reasoning_segments_json": null,
  "metrics_json": "{\"tokenCount\":10,\"promptTokens\":593,\"cachedTokens\":524,\"cacheDetail\":\"paged+tq-native\",...}"
}
```

This is a PASS for the narrow post-guard DB condition: no content-identical
reasoning was persisted for that live UI turn.

## Live q4 / cache evidence

Evidence file:

- `laguna-health-after-railguard-3431c86b3.json`

Relevant health fields:

- `kv_cache_quantization.enabled=true`
- `kv_cache_quantization.mode="turboquant-storage"`
- `kv_cache_quantization.bits=4`
- `kv_cache_quantization.stored_prefix_quantization="turboquant-q4"`
- `kv_cache_quantization.auto_policy="mixed_swa_full_attention_kv_storage_tq4"`
- `scheduler.cache_hit_tokens_by_detail` includes:
  - `paged+disk+tq-native`
  - `paged+tq-native`
- `scheduler.last_cache_execution.cache_detail="paged+tq-native"`
- `scheduler.last_cache_execution.cached_tokens=524`
- `block_disk_cache.tq_native_enabled=true`
- `block_disk_cache.tq_native_writes=16`
- `block_disk_cache.tq_native_hits=5`

This is current live telemetry for q4 TurboQuant storage and paged/L2 cache use.
It is not yet an exhaustive proof for disk-only mode, cross-chat partial-block
restore, eviction, or restart-after-eviction behavior.

## API evidence

Evidence file:

- `laguna-responses-rail-dup-fail-ad931a45f.json`

Raw `/v1/responses` SSE for a one-message prompt produced:

- content joined: `LAGUNA API OK`
- reasoning joined: empty string
- terminal event: `response.completed`

This supports that the duplicate reasoning observed in one UI history row was
not emitted by that simple raw Responses request. It does not prove every API
shape or full-history shape.

## Retained negative / partial evidence

Do not hide these:

- DB row `60` before `3431c86b3` persisted `content="LAGUNA OK"` and
  `reasoning_content="LAGUNA OK"` with `reasoning_segments_json=["LAGUNA OK"]`.
  That is the bug class fixed by `3431c86b3`.
- An older Laguna row in the same profile produced a long garbage
  `reasoning_content` and `[Generation interrupted]`; that remains retained
  negative history, not a current pass.
- A required-tool Laguna row did complete with a real `file_info` call and
  `LAG-S21-UI-TOOL-V9-RESTART-DONE SIZE=5.2 KB`, but that row predates the
  final rail-guard commit and does not close every agentic protocol surface.
- New-chat creation was flaky through the `uidrv-once.cjs` text-click helper
  in this turn; the proof row was in the active Laguna chat. Do not claim a
  fresh-chat UI proof from this row.

## Still open before a release

- Full Python suite.
- Full panel suite.
- Production Electron build.
- Bundle-python rebuild if engine changes are included.
- Signed/notarized/stapled Sequoia and Tahoe DMGs.
- Public updater/GitHub/PyPI/Homebrew/version truth reconciliation.
- Fresh-chat UI rail proof.
- Raw Chat Completions, Responses message-array/full-history, Anthropic,
  Ollama reasoning/tool loop proof.
- Laguna disk-only block L2 partial-prefix reuse with paged RAM off.
- Cross-chat/cross-session partial-prefix SSD reuse and restart proof.
- Eviction and old-block invalidation proof after `1c8ef8ecd`.
- Broader model-family rail proof for Qwen/Bonsai/Gemma/MiniMax/OpenPangu/DSV4.
- JANG/JANGTQ quantization pipeline and separate `jangtools` repo sync.

Verdict: **PARTIAL_NO_RELEASE**. The current source and live evidence support
only the scoped Laguna rail-guard/cache/startup rows above.

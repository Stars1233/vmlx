# Model-derived Chat Settings and sampler override gate

Date: 2026-07-20/21 PDT
Source tree: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
Base commit: `7bdb3a0181c068a73b4b84d299239292c42321fa`
Live app: Electron dev source, profile `/Users/eric/.vmlx-v1613-responsive-dev`, CDP `9335`
Live model: `/Volumes/EricsLLMDrive/jangq-ai/gemma-4-12B-it-qat-JANG_4M`, UI port `8141`

## Verdict

- `PASS-LIVE-SCOPED`: the Gemma affine-JANG bundle-to-session-to-drawer-to-Responses-payload-to-engine chain is current-source proven.
- `PASS-LIVE-SCOPED`: an explicit Chat Settings `Top K = Off` survives Save, SQLite, request serialization, and engine resolution. Before the fix the UI stored `0`, but the request builder omitted it and the engine silently re-inherited bundle `64`.
- `PASS-LIVE-SCOPED`: Reset clears sampler overrides to SQL `NULL`, the drawer immediately returns to current bundle values (`temperature=1.00`, `top_p=0.95`, `top_k=64`, neutral repetition penalty), and the next Electron request omits sampler overrides so the engine resolves the bundle values.
- `PASS-LIVE-SCOPED`: raw streaming Responses, Chat Completions, and Ollama chat forward explicit `top_k=0`, produce nine progressive content deltas each, and finish with the requested exact marker and a valid terminal event.
- `PASS-SOURCE`: no family may receive a hidden engine-only top-k fallback. Request, explicit startup/session CLI, or bundle metadata are now the only non-zero sources. The existing Ling/Bailing `20` fallback was removed because it disagreed with the UI and was not bundle-grounded.
- `PARTIAL`: this gate has not live-loaded a locally unavailable Ling/Bailing artifact, and has not yet repeated the full visual chain on JANGTQ/MXTQ, base MLX/MXFP, DSV4/M3 native-cache, or a bundle with a non-neutral repetition penalty. Those rows stay open and should be folded into each family’s next live gate.

## Source trace

- `panel/src/renderer/src/components/chat/ChatSettings.tsx`
  - Top K `0` and repetition penalty `1.0` remain explicit overrides.
  - Reset removes model-owned sampler fields instead of persisting a stale copy of current bundle values.
- `panel/src/shared/chatSettingsResetPolicy.ts`
  - Reset preserves only tool/workspace ergonomics; sampler/output/reasoning fields return to inheritance.
- `panel/src/main/ipc/chat.ts`
  - Chat Completions and Responses serialize `top_k` whenever the override is non-null, including `0`.
- `panel/src/main/api-gateway.ts`
  - Ollama chat and generate forward non-negative `top_k`, including `0`; negative sentinels remain omitted.
- `vmlx_engine/server.py`
  - `_resolve_top_k` remains request > explicit startup/session > bundle > disabled.
  - The engine-only Ling/Bailing fallback was removed.

## Live Electron proof

1. The dev shell was fully restarted so main-process request-builder changes—not just renderer HMR—were active.
2. Gemma was started with the real Sessions `Start` button. The session showed `Running`, port `8141`, PID `8484`, and the startup log contained `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`.
3. With `Top K = Off`, SQLite held `top_k=0`. The live turn returned exact `SAMP-TOPK0-LIVE3-DONE` with no reasoning or warning. The request diagnostic contained `"top_k":0`; resolved engine kwargs contained temperature/top-p but no top-k, which is the engine representation of disabled top-k.
4. The same turn restored `164/210` prompt tokens as `paged+mixed_swa+disk+tq-native`, TTFT `0.46 s`, total `0.9 s`.
5. Real Reset changed the range control from `0` to bundle `64` and cleared all sampler SQL columns. With Thinking Off reselected independently, the next exact turn returned `SAMP-INHERIT64-LIVE-DONE`; the request omitted `top_k`, and engine kwargs resolved `top_k:64` from the bundle.
6. The inherited turn restored `209/257` prompt tokens as `paged+mixed_swa+tq-native`, TTFT `0.33 s`, total `0.6 s`.

Screenshots:

- `sampling-topk-off-postfix.png`: real drawer visibly holds `Top K Off` instead of snapping back to 64.
- `sampling-topk0-live3-pass.png`: real Electron exact output and cache/timing surface.
- `sampling-inherit64-drawer-visible.png`: post-Reset drawer visibly shows Top K 64, Min P 0.00, repetition penalty 1.00, and model-default Max Tokens.
- `sampling-inherit64-live-pass.png`: post-Reset exact output and cache/timing surface.

Structured state and logs:

- `model-session-db-parity.json`: bundle values, persisted session detection, cleared chat override, and Electron message row.
- `live-request-resolution.log`: before/after request shapes and resolved kwargs.
- `health-after-ui-start.json`: current engine/cache health after the real Start-button load.

## Raw API/gateway streaming proof

- `responses-topk0.sse` / `.timed.sse`
  - nine `response.output_text.delta` events
  - exact `SAMP-API-RESP0-DONE`
  - `response.output_text.done` and `response.completed`
- `chat-topk0.sse` / `.timed.sse`
  - nine non-empty content deltas
  - exact `SAMP-API-CHAT0-DONE`
  - terminal `finish_reason=stop` and `[DONE]`
- `ollama-topk0.ndjson` / `.timed.ndjson`
  - nine non-empty message-content chunks
  - exact `SAMP-OLLAMA0-DONE`
  - terminal `done=true`
- Engine resolution for all three explicit-zero routes omitted top-k; the Ollama route also preserved explicit neutral `repetition_penalty:1.0`.

## Tests

- Panel focused suite: 5 files, `519 passed`.
- Panel typecheck: `tsc --noEmit` passed.
- Engine sampling-focused audit: `51 passed`, `539 deselected`.
- Generation-default cross-matrix contract: `status=pass`:
  - panel generation defaults: 28 passed
  - engine generation defaults: 61 passed
  - local metadata audit: 6 passed, 1 skipped
  - panel CLI startup contract: 22 passed
- `git diff --check` is required again immediately before the scoped commit.

## Retained observations

- The Electron startup log identifies the app as `v1.6.14` but reports the PATH engine as `1.6.12`. This gate does not classify that version-string mismatch; the release/version-truth row must trace installed package metadata separately.
- Repetition penalty `1.0` now survives UI/gateway serialization, but a live bundle with a declared value above `1.0` is still required to prove that the neutral UI override disables the bundle value.
- A Ling/Bailing model bundle was not found in the active model roots, so removal of the hidden family fallback is source/test proven only.

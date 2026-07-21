# Nemotron Omni current image/video streaming and session-cache gate

Date: 2026-07-20

Host: `erics-m5-max.local`

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Base HEAD: `28bf22729824c6265dd988a8e744308b13bfae30`

## Current verdict

- Real Electron Start-button eager load and one-model ownership: **PASS-LIVE
  scoped**. PID 22226 was the only engine process, the previous Gemma process
  was gone, `/health` reported `model_loaded=true` and
  `last_request_time=null`, and the UI visibly showed the selected Nemotron
  session plus Stop.
- Bundle/runtime classification: **PASS-SOURCE+LIVE scoped**. The artifact is
  `weight_format=mxtq`, JANGTQ2 Hadamard/codebook weights, not affine JANG and
  not base MLX MXFP. Runtime cache state is hybrid: q4 TurboQuant only for six
  attention-KV layers, native/full-precision SSM companion state elsewhere,
  with async rederive on a miss or warm pass.
- Same-image in-process continuation: **PASS-LIVE scoped**. The UI first read
  `vMLX` from a real PNG. A no-attachment second turn recalled it with prompt
  TTFT reduced from 7.34 s to 0.37 s; logs recorded `continuing conversation
  (prefix matches)`, zero new images, and a new q4-KV/native-SSM snapshot.
- Different-image salt isolation: **PASS-LIVE scoped for the patched Auto
  turn**. A fresh chat attached a different image whose unseen code was
  `BANANA8426`; the engine logged a cache reset and did not leak `vMLX`.
- Omni omitted-Max-Tokens propagation: **PASS-SOURCE+LIVE scoped after fix**.
  The pre-fix fresh Auto turn stopped at exactly 256 tokens with 849 reasoning
  characters, empty visible content, and truncation warnings even though the
  bundle/UI default was 16,384. Current Chat, Responses, and Anthropic Omni
  entries forward their already-resolved max-token, temperature, and top-p
  values to the bridge. The patched Electron Auto turn emitted separate
  reasoning plus exact visible content in 107 tokens with no warning.
  Omitted-max raw Responses emitted 257 reasoning and 24 content deltas plus
  one completed event; raw Chat emitted 351 reasoning and 23 content deltas,
  stop, usage, and one `[DONE]`. Both exact-finaled the unseen image code.
- Video streaming and process-restart session L2: **PASS-LIVE for transport,
  stream structure, and cache restore; PARTIAL for strict output format**. A
  fresh Electron MP4 turn produced 33 distinct visible states, 785 characters
  of separate reasoning, exact `BANANA8426`, and the requested completion
  marker. After real Stop/Start changed PID 23620 to PID 24498, health was
  loaded with `last_request_time=null`. The same-chat no-attachment follow-up
  restored the 51,847,398-byte q4-attention-KV/native-SSM snapshot in 0.000328
  seconds (`hits=1`, `misses=0`) and streamed exact recall, but appended an
  unrequested Python block after the two requested lines. That format miss is
  retained rather than normalized away.
- Raw Responses MP4 with all max fields omitted: **PASS-LIVE scoped**. It
  emitted 447 reasoning deltas, 25 content deltas, one text-done, and one
  completed event, then exact-finaled `BANANA8426` plus the requested marker.
- The implementation exposes one `latest.safetensors` architecture snapshot.
  Latest-snapshot replacement semantics, bounded/multi-snapshot eviction, and
  partial-prefix matching across different media histories remain **OPEN**.

## Root cause and repair

`create_response()` resolved the omitted panel/API output cap to the bundle's
`generation_config.json::max_new_tokens=16384` in `chat_kwargs`, then rebuilt
the internal Chat request from the unresolved `request.max_output_tokens`.
`dispatch_omni_chat_completion()` consequently applied its historical `256`
fallback. Chat Completions dispatched before its normal `chat_kwargs`
resolution and had the same omitted-value risk; the Anthropic adapter is kept
on the same explicit resolved contract for parity.

Current source adds explicit `effective_max_tokens`,
`effective_temperature`, and `effective_top_p` inputs to the Omni bridge.
Each protocol handler supplies the same resolved values used by its ordinary
generation route. No hidden sampler clamp, prompt coercion, answer rewrite, or
model-specific synthesis pass was added.

## Live output-emission evidence

- `ui-image-a-row.json`: exact `NEMO-IMG-A1-DONE VALUE=vMLX`, null reasoning,
  no warning/tool state.
- `ui-image-a-dom-trace.json`: character-level progressive UI paint from `N`
  through the exact final after the real image prefill.
- `ui-image-a2-row.json` and `ui-log-after-image-a2.png`: no-attachment exact
  recall plus the visible owning `prefix matches` log.
- `ui-image-b-row.json`: retained pre-fix failure; 256 reasoning-only tokens,
  empty content, and explicit length warnings.
- `ui-image-b-postfix-row.json`: patched Auto result with 284 reasoning chars,
  exact two-line visible output, no warning/tool state.
- `raw-responses-omitmax-auto.sse`: omitted-max Responses proof.
- `raw-chat-omitmax-auto.sse`: omitted-max Chat Completions proof.
- `raw-responses-video-omitmax-auto.sse`: raw omitted-max MP4 Responses proof
  with 447 reasoning and 25 content deltas plus one completed event.
- `health-before-request.json`: eager load, exact hybrid cache layout, zero
  prior request.
- `health-after-ui-image-a.json` and `health-after-ui-image-a2.json`: 50.97 MB
  then 51.09 MB architecture snapshots with q4 attention KV/native SSM.
- `ui-video-a1-dom-trace.json`, `ui-video-a1-row.json`, and paired screenshots:
  fresh Electron MP4 reasoning/content progression and exact video read.
- `health-after-video-restart-before-request.json`: replacement process loaded
  before any request after the real UI restart.
- `ui-video-a2-l2-row.json`, `ui-video-a2-l2-dom-trace.json`, and
  `health-after-ui-video-a2-l2.json`: no-attachment restart follow-up, the
  exact q4-KV/native-SSM restore counters, and the retained format miss.
- `ui-log-after-video-a2-l2.png`: visible owning log with the restore,
  prefix-match, and re-persist events.

The deterministic media fixtures and hashes are retained as
`image-a-vmlx.png`, `image-b-banana8426.png`, and
`video-a-banana8426.mp4`. `omni_omitted_max_probe.py` records the raw omitted-
cap requests instead of relying on transcript reconstruction.

## Validation at this scoped repair

- `py_compile` for `server.py` and `omni_multimodal.py`: passed.
- Focused Omni/server audit selection: 30 passed, 580 deselected.
- `git diff --check`: passed.

Full Python/panel suites are not claimed by this scoped commit; they remain a
separate checkpoint gate.

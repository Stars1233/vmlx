# Gemma 4 26B model-derived sampling parity

Date: 2026-07-21

Host: `erics-m5-max.local`

Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Source: `1f42f46e5e97a596ea68fd1a2af6bdb8d329c16c`

## Verdict

`PASS-LIVE_SCOPED` for model-owned generation settings on the real affine
`JANG_4M` Gemma 4 26B-A4B MoE artifact. This gate does not promote JANGTQ/MXTQ,
base MLX/MXFP, DSV4/M3 typed routes, or a non-neutral repetition-penalty
artifact.

| Axis | Verdict | Evidence |
|---|---|---|
| Artifact identity | PASS-SOURCE | The bundle is affine `JANG_4M` (`mx.quantize`, measured 4.26 bits), not JANGTQ/MXTQ Hadamard-codebook and not base MLX MXFP. |
| Bundle defaults | PASS-SOURCE | `generation_config.json` declares sampling on, temperature `1.0`, top-p `0.95`, and top-k `64`. It does not declare min-p or repetition penalty, so the UI correctly displays neutral `0.00` and `1.00`. |
| Session detection | PASS-LIVE | SQLite session config stores `defaultTemperature=100`, `defaultTopP=95`, `defaultTopK=64`, `defaultMinP=0`, `defaultRepetitionPenalty=0`, and `defaultSamplingDefaultsDeclared=true`. |
| Visible drawer parity | PASS-LIVE | The real Electron Chat Settings drawer at CDP 9335 visibly showed `1.00`, `0.95`, `64`, `0.00`, and `1.00`. Thinking Off and built-in tools Off were saved before the decisive UI2 turn. |
| Request inheritance | PASS-LIVE | UI2 `CHAT_DIAG` sent no sampling overrides and did send `enable_thinking=false`, `thinking_mode=instruct`, and `has_tools=false`. This is the expected inheritance contract, not a hidden copied override. |
| Effective runtime | PASS-LIVE | The engine resolved temperature `1.0`, top-p `0.95`, top-k `64`, max tokens `16384`, and thinking Off from that request/bundle chain. |
| Electron output | PASS-LIVE | UI2 visibly painted an intermediate `G4-26-SA` state and exact-finaled `G4-26-SAMP-UI2-DONE`. SQLite row 281 has non-empty content, no reasoning, no tool calls, and no warning. |
| Active cache/quant truth | PASS-LIVE scoped | Health identifies affine JANG 4.26, Gemma mixed-SWA storage-only TQ4, paged RAM, and block-disk L2. This short settings turn was not a cache-hit gate and makes no restore claim. |

## Source trace

- `panel/src/main/sessions.ts::readBundleStartupDefaults` and
  `applyBundleStartupDefaults` read bundle generation defaults and persist the
  normalized session fields.
- `panel/src/renderer/src/components/chat/ChatSettings.tsx` renders the saved
  override when present and otherwise renders the model default for each
  sampler control.
- `panel/src/main/ipc/chat.ts::buildRequestBody` omits absent sampler overrides,
  preserves explicit zero/neutral overrides, and forwards explicit values to
  both Responses and Chat Completions.
- `vmlx_engine/server.py::_bundle_sampling_default`, `_resolve_temperature`,
  `_resolve_top_p`, `_resolve_top_k`, and `_log_resolved_sampling_kwargs` apply
  JANG metadata / generation-config precedence and log the effective kwargs.

The traced readers, renderer, IPC request builder, and engine resolver all have
production call sites. No dead/test-only compatibility branch or model-specific
output rewrite was added for this closure.

## Negative control retained

UI1 is not counted as the saved Thinking-Off/tools-Off proof. The drawer state
was changed but not saved before it closed; the log truthfully shows
`has_tools=true` and `enable_thinking=true`. UI1 still independently showed
that omitted sampling overrides resolve to the bundle values. UI2 clicked Save,
then ran the decisive request with the expected false flags.

## Evidence

- `g4-26-samp-ui2-settings-final.png` — completed UI2 chat with the saved real
  drawer values visible.
- `g4-26-samp-ui2-result.png` — exact completed UI2 turn and rendered metrics.
- `g4-26-samp-ui2-trace.json` — three observed Electron states, including a
  partial visible content paint before completion.
- `bundle-generation-config.json` — exact relevant fields from the artifact.
- `session-config-snapshot.json` — normalized persisted session defaults and
  launch controls.
- `message-row-281.json` — terminal assistant row.
- `runtime-log-excerpt.txt` — UI1 negative-control and UI2 decisive request /
  resolved-sampler lines.
- `g4-26-health-current.json` — live running engine/cache/quant snapshot.

## Remaining boundaries

- Retain family breadth for JANGTQ/MXTQ, base MLX/MXFP, DSV4/M3 typed routes,
  and a bundle that declares a non-neutral repetition penalty.
- Retain Gemma Auto-thinking economy/attachment-awareness and strict-format
  quality as separate PARTIAL rows.
- Do not rerun the already-proven generic Chat/Responses/Ollama explicit-zero
  forwarding gate for this same affine family.

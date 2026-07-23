# Qwen 3.6 27B bundle-default settings parity

Date: 2026-07-23 (America/Los_Angeles)

Status: `FIXED + VERIFIED-LIVE-SCOPED / Q27 FAMILY GATE PARTIAL`.

## Exact boundary

- Checkout:
  `/Users/eric/mlx/vllm-mlx-r17-consolidation`
- Branch:
  `codex/v1.6.17-consolidation-20260723`
- Bundle:
  `/Volumes/EricsLLMDrive/dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP`
- Real Electron profile:
  `/Users/eric/.vmlx-r17-consolidation-dev`
- CDP `9335`, direct port `8009`, gateway `8090`
- Save & Restart PID: `19209`

Bundle file hashes are retained in `r17-q27-bundle-sha256.txt`.

## Root cause and change

The bundle declares `temperature=1.0`, `top_p=0.95`, and `top_k=20`.
`jang_config.json` preserves native MTP but does not replace those sampling
defaults.

Fresh native-MTP sessions nevertheless defaulted
`nativeMtpMode=deterministic`. The launcher therefore added
`--native-mtp-sampling-policy deterministic-defaults`, and the engine installed
greedy omitted-request defaults `0/1/0/0`. The Chat Settings renderer mirrored
that real server policy. SQLite did not persist sampler overrides; the wrong
owner was the shared startup-policy default.

- Fresh and missing native-MTP mode now defaults to `auto`.
- Auto launches `compatible-only`; generation-config/JANG sampling remains
  authoritative for omitted app, direct API, and gateway requests.
- An explicitly saved Deterministic override still displays and applies
  `0/1/Off`, because that user-owned server setting really changes omitted
  requests.
- Process adoption now records the live MTP sampling policy from `/health`,
  with argv as fallback. A deterministic orphan cannot be adopted as Auto and
  display bundle values while its process uses greedy defaults.
- No hidden sampler clamp, prompt coercion, output rewrite, or model-specific
  exception was added.

## Current live proof

1. Real Electron Server Settings selected `Auto (bundle defaults)` and used
   Save & Restart. PID `19209` launched with
   `--native-mtp-depth 3 --native-mtp-sampling-policy compatible-only`.
2. The real Chat Settings drawer visibly showed Temperature `1.00`, Top P
   `0.95`, Top K `20`, Min P `0.00`, Repetition Penalty `1.00`, Thinking Auto.
3. SQLite retained `NULL` for temperature, top-p, top-k, min-p,
   repetition-penalty, max-tokens, and enable-thinking. Session metadata
   separately retained model-derived `100/95/20` plus `nativeMtpMode=auto`.
4. Direct and gateway `/v1/capabilities` both reported bundle and effective
   omitted-request defaults `1.0/0.95/20`, min-p `0`, with MTP policy
   `compatible-only`.
5. Direct Chat without sampler overrides emitted 13 progressive content
   deltas and exact-finaled `R17-Q27-DIRECT-DEFAULTS-OK`.
6. Gateway Chat without sampler overrides emitted 14 progressive content
   deltas and exact-finaled `R17-Q27-GATEWAY-DEFAULTS-OK`.
7. Both streams ended with one `finish_reason=stop` and one `[DONE]`.

## Focused source verification

- Six panel test files: `411/411` passed.
- TypeScript: `tsc --noEmit` passed.
- `git diff --check` passed.

Coverage includes standard generation config, JANG field precedence,
first-session hydration, chat override omission, Reset, request serialization,
explicit deterministic override truth, and process-adoption policy retention.

## Retained artifacts

- `r17-q27-chat-bundle-defaults-live.png`
- `r17-q27-mtp-auto-save-restart.png`
- `r17-q27-chat-sqlite-defaults.json`
- `r17-q27-session-sqlite-defaults.json`
- `r17-q27-direct-capabilities.json`
- `r17-q27-gateway-capabilities.json`
- `r17-q27-health-auto-bundle.json`
- `r17-q27-health-after-default-api.json`
- `r17-q27-direct-defaults.sse`
- `r17-q27-gateway-defaults.sse`
- `r17-q27-bundle-sha256.txt`

## Still open

- Q27 three-turn Electron reasoning/tool/history/KaTeX.
- Q27 Responses, Anthropic, and Ollama agentic protocol rows.
- Q27 hybrid SSM/GDN q4 attention-KV partial SSD reuse with Paged On and Off,
  restart refault, and cross-chat/session reuse.
- Q27 image/video rows.
- Historical sessions already saved as Deterministic remain explicit
  deterministic sessions; they are not silently migrated. Selecting Auto once
  persists on subsequent restarts.
- Full suites, production build, bundled-engine provenance, signed installed
  app, packaging, notarization, tagging, and publication.

Overall v1.6.17 remains `PARTIAL / NOT RELEASE-READY`.

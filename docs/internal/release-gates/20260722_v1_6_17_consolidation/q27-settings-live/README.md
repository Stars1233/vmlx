# Qwen 3.6 27B bundle-default settings parity

Date: 2026-07-23 (America/Los_Angeles)

Status:
`SETTINGS + REASONING/TOOLS/PROTOCOL + HYBRID SSD CACHE VERIFIED-LIVE /
MEDIA + EVICTION OPEN`.

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
- Settings Save & Restart PID: `19209`
- Cache proof PIDs:
  - Paged On restart from SSD: `21624`
  - Paged Off + Block SSD: `23122`
  - Paged Off real Stop/Start: `23461`
  - Paged On restored: `23696`

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

## Three-turn Electron proof

The same real Q27 Electron session completed three inspected turns:

1. A no-tool math/currency turn stored `2,989` reasoning characters separately
   from the exact two-line visible answer. The live renderer contained two
   KaTeX nodes and no math-fallback node or raw `\times`.
2. A required-tool turn emitted exactly one schema-valid
   `file_info(path="panel/package.json")` call, executed the real tool once,
   preserved the result (`5.2 KB`), then exact-finaled
   `Q27-UI-TOOL-DONE SIZE=5.2 KB`.
3. A no-tool follow-up recalled the exact prior path and size and privately
   computed `7 * 8`, exact-finaling the requested two-line result.

All three turns had non-empty reasoning (`2,989`, `679`, and `1,565`
characters). Their SHA-256 hashes are all different, ruling out byte-identical
stale reasoning replay. Reasoning remained in the reasoning rail; visible
content contained no parser/tool control markup.

Retained proof:

- `r17-q27-ui-turn1.png`
- `r17-q27-ui-turn2.png`
- `r17-q27-ui-turn3.png`
- `r17-q27-ui-messages.json`
- `r17-q27-ui-turn*-progress.json`
- `r17-q27-ui-turn*-body.html`

## Direct and gateway streaming protocol proof

`r17-q27-agentic-protocol-stream.json` retains a three-round real tool
continuation for direct and Electron-gateway versions of:

- OpenAI Chat Completions;
- OpenAI Responses;
- Anthropic Messages;
- Ollama Chat.

Each of the eight flows:

- streamed private reasoning separately from visible content;
- emitted exactly one `file_info(panel/package.json)` call in round one;
- emitted exactly one `run_command(pwd)` call in round two;
- received the real tool results and exact-finaled in round three;
- kept tool rounds free of visible prose/control markup;
- used progressive final content and truthful route-specific terminal events;
- reported no protocol/stream errors or stale/duplicated reasoning.

This artifact is streaming-only. It does not replace the existing source and
focused-test coverage for non-stream normalization, and abort recovery was
intentionally not rerun in this Q27 row.

## Hybrid SSM/GDN q4-TQ block-L2 proof

The live bundle resolved `cache_type=hybrid_ssm_v1`: 16 attention-KV layers
use q4 TurboQuant block storage while 48 SSM/GDN companion states remain typed
native full precision.

- The cold `9,498`-token request needed a full prefill and exact-finaled.
- A first changed-tail attempt found a `9,408`-token KV boundary without a
  matching typed companion and correctly revoked the hit instead of using
  incomplete state. After the required checkpoint was stored, aligned
  changed-tail requests reused `9,408` tokens and prefetched only
  `101-105` tokens.
- The suffix-only negative reused no interior shared corpus. This is correct:
  KV prefix caching reuses the longest continuous prefix; it must not treat an
  arbitrary interior/suffix token run as a safe attention-state hit.
- With Paged RAM On, a real Electron Stop/Start loaded PID `21624` from zero
  scheduler/L1 state. The next request restored `9,408/9,509` tokens from 147
  SSD blocks plus typed SSM state as `paged+ssm+disk+tq-native`.
- The real Server Settings UI turned In-Memory Paged Cache Off while leaving
  Block Disk Cache checked and enabled. Save & Restart loaded PID `23122` with
  `--no-paged-cache`, `backend_mode=block_disk_only`, zero RAM/L1 cache, and
  retained attention plus SSM data on SSD. It restored `9,408/9,510` tokens as
  `block-disk+ssm+tq-native`.
- A second real UI Stop/Start loaded PID `23461`, again with zero RAM state.
  It restored `9,408/9,513` tokens from the same 147 SSD blocks and typed
  companion state.
- The UI restored Paged RAM On without disabling Block SSD. PID `23696`
  started with zero RAM cache, retained the SSD stores, and restored
  `9,408/9,512` tokens as `paged+ssm+disk+tq-native`.
- Every hit exact-finaled with `finish_reason=stop` plus `[DONE]`; all disk
  rows recorded 147 q4-TQ native block hits.

Retained proof:

- `r17-q27-l2-*.json`
- `r17-q27-health-*-before-request.json`
- `r17-q27-paged-off-ssd-on-before-save.png`
- `r17-q27-paged-off-stopped-2.png`
- `r17-q27-restore-paged-on-before-save.png`

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
- `r17-q27-agentic-protocol-stream.json`
- `r17-q27-ui-messages.json`
- `r17-q27-ui-turn*.png`
- `r17-q27-l2-*.json`
- `r17-q27-health-*.json`
- `r17-q27-paged-off-*.png`
- `r17-q27-restore-paged-on-before-save.png`

## Still open

- Q27 image/video rows.
- Q27 block-disk capacity eviction/refault and corrupt/missing typed-companion
  fallback.
- Signed-installed-app repetition.
- Historical sessions already saved as Deterministic remain explicit
  deterministic sessions; they are not silently migrated. Selecting Auto once
  persists on subsequent restarts.
- Full suites, production build, bundled-engine provenance, signed installed
  app, packaging, notarization, tagging, and publication.

Overall v1.6.17 remains `PARTIAL / NOT RELEASE-READY`.

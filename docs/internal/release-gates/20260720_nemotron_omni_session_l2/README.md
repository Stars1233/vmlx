# Nemotron Omni process-restart session-L2 gate

Date: 2026-07-20

Host: `erics-m5-max.local`

Checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Base HEAD before this scoped change: `edc03bcde88e06624c085a72614157c9759fc6a9`

## Verdict

- Nemotron Omni process-restart restoration of its architecture-owned mixed
  attention-KV plus SSM session: **PASS-LIVE scoped on current post-v1.6.14
  source**.
- Runtime storage policy: **PASS-LIVE scoped**. Attention KV is persisted with
  MLX quantized KV at q4; Mamba/SSM `ArraysCache` state remains native/full
  precision. This is not an ordinary scheduler-cache claim.
- Real Electron output emission after restoration: **PASS-LIVE scoped**.
  Reasoning painted progressively in the reasoning rail and visible content
  painted separately before the terminal metrics row.
- Raw Chat Completions streaming after a second process restart:
  **PASS-LIVE scoped**.
- Explicit Block Disk Cache (L2) Off: **PASS-LIVE negative control**. The
  persisted file remained on disk, but the current process did not restore or
  rewrite it. The UI was restored to On afterward.
- Broader cache breadth: **PARTIAL**. This implementation persists the latest
  exact conversation/media prefix for one Omni bundle. Multi-snapshot LRU,
  partial-prefix/block reuse inside the architecture-owned session file,
  bounded eviction, image/video restart controls, and signed-app repetition
  remain separate open rows.
- Public checkpoint: **NOT INCLUDED**. Tagged/public v1.6.14 predates this
  change and correctly retains the row as open at tagged source.

## Bundle-grounded route

`bundle-facts.json` was read from the live bundle
`dealignai/Nemotron-Omni-Nano-JANGTQ-CRACK` on the external volume:

- `config.json`: `model_type=nemotron_h`, `weight_format=mxtq`,
  `mxtq_bits=2`;
- `jang_config.json`: `profile=JANGTQ2`, Hadamard/codebook MXTQ with 2-bit
  routed experts and 8-bit attention/shared/Mamba/embed/head weights;
- `config_omni.json`: `NemotronH_Nano_Omni_Reasoning_V3`.

The artifact is JANGTQ/MXTQ. It is not affine JANG and not base MLX MXFP.
The q4 value in this gate refers only to persisted runtime attention KV.

## Root cause and source repair

The Omni dispatcher owns a persistent `OmniSession` containing mixed attention
KV and Mamba/SSM state. Process restart discarded that state, while ordinary
scheduler paged/TQ/L2 counters described a different cache and therefore could
not prove media-conditioned Omni restoration.

Current source adds an architecture-specific `nemotron_omni_session_v1` path:

- `vmlx_engine/omni_multimodal.py`
  - fingerprints the resolved bundle path plus `config.json`,
    `config_omni.json`, `jang_config.json`, and the safetensor index;
  - persists the exact media/user-turn signature and text history;
  - converts only attention `*` cache entries to MLX q4 quantized KV;
  - leaves Mamba/SSM `ArraysCache` entries native;
  - atomically replaces the latest snapshot;
  - restores only when schema, exact bundle fingerprint, cache topology, and
    exact conversation/media prefix all match;
  - queues persistence behind decode after the stream's done event has been
    enqueued, so snapshot writing cannot withhold visible terminal output.
- `vmlx_engine/server.py`
  - reads the effective loaded scheduler's Block Disk Cache toggle;
  - forwards it through Chat, Responses, and Anthropic Omni dispatch;
  - reports architecture-specific `omni_multimodal.session_l2` health rather
    than promoting ordinary scheduler counters.
- `tests/test_omni_multimodal.py`
  - round-trips a real MLX q4 `QuantizedKVCache` plus native `ArraysCache`;
  - rejects a different media prefix;
  - requires one persistence schedule after a successful streamed turn.

The same scoped commit carries the source and regression tests alongside this
evidence; no transcript-only source claim is used.

## Current-source Electron proof

The exact current source was relaunched through the real Electron UI on CDP
9335. Before each restoration request, `/health` reported
`model_loaded=true` and `last_request_time=null`; the model was loaded before
generation.

1. Seed turn on PID 82562:
   - prompt introduced restart-only code `FIR-9928`;
   - exact visible answer `SEEDED`;
   - separate 297-character reasoning;
   - health recorded one snapshot store, q4 attention KV, native SSM, no error.
2. Real UI Stop followed by real UI Start produced fresh PID 82724.
3. Post-restart turn:
   - exact visible answer `FIR-9928`;
   - separate 396-character reasoning, distinct from the seed reasoning;
   - observer event 1 showed `Waiting for model response...` at 8 ms;
   - reasoning grew across 14 sampled paints;
   - visible content began as `FIR` at 5,513 ms;
   - exact final arrived at 5,616 ms;
   - SQLite row 106 contains non-empty `content=FIR-9928`, separate reasoning,
     metrics, no tool calls, and no warnings.

After that turn, `nemo-l2-current-hit-health.json` reports:

- `hits=1`, `misses=0`, `stores=1`;
- `last_restore_seconds=0.000317`;
- `last_store_seconds=0.011656`;
- `attention_codec=mlx-quantized-kv-q4`;
- `ssm_codec=native-arrays`;
- `last_error=null`, `pending=0`.

`current-source-omni-log-excerpt.txt` independently records the q4-KV/native-
SSM restore, prefix match, and subsequent atomic persistence. The first-paint
and final screenshots show the real Electron session, exact model/PID,
separate reasoning rail, non-empty answer, and effective Paged KV UI state.

No tool was requested in this scoped cache-restoration row; tool-loop behavior
is neither promoted nor downgraded by this evidence.

## Current-source raw API proof

After another real UI Stop/Start, `nemo-l2-current-raw-start-health.json`
again recorded a loaded, request-cold process. A raw `curl -N` request to
`/v1/chat/completions` used the Electron conversation history and asked for the
remembered audio marker plus restart-only code.

`nemo-l2-current-raw-chat-stream.sse.gz` contains the byte-preserved raw SSE
capture. After decompression it contains:

- 129 non-empty `reasoning_content` deltas;
- 13 non-empty visible `content` deltas;
- final visible `blue6813 FIR-9928`;
- exactly one `finish_reason=stop`;
- exactly one usage object;
- exactly one `[DONE]`.

The post-request health snapshot records another real architecture-session hit
(`hits=1`, `misses=0`), q4/native codecs, 0.003711-second restore,
0.021615-second store, and no error.

## Explicit-Off negative control

The real Server Settings drawer was used to uncheck Block Disk Cache (L2) and
apply Save & Restart. Before any request, health reported `enabled=false` even
though the existing snapshot file remained present. The Electron turn
`[NEMO-L2-EXPLICIT-OFF]` progressively emitted exact non-empty
`OFF-PATH-ACTIVE`; health retained `hits=0`, `stores=0`, and no restore/persist
log appeared. `nemo-l2-explicit-off-final.png` visibly shows the unchecked L2
control.

The same real UI then re-enabled L2 and applied Save & Restart.
`nemo-l2-final-on-settings.png` and `nemo-l2-final-on-health.json` preserve the
final effective On state.

## Validation

- Expanded Omni/multimodal selection: 45 passed, 563 deselected.
- Complete `tests/test_server.py`: 119 passed, 3 deselected.
- `git diff --check`: passed before staging.

JUnit reports are `nemo-l2-omni-tests.xml` and
`nemo-l2-server-tests.xml`.

## Evidence index

- `bundle-facts.json`
- `electron-current-source-db.json`
- `current-source-omni-log-excerpt.txt`
- `nemo-l2-current-seed-start-health.json`
- `nemo-l2-current-seed-health.json`
- `nemo-l2-current-seed-stream.json`
- `nemo-l2-current-seed-first-paint.png`
- `nemo-l2-current-seed-final.png`
- `nemo-l2-current-hit-start-health.json`
- `nemo-l2-current-hit-health.json`
- `nemo-l2-current-hit-stream.json`
- `nemo-l2-current-hit-first-paint.png`
- `nemo-l2-current-hit-final.png`
- `nemo-l2-current-raw-start-health.json`
- `nemo-l2-current-raw-payload.json`
- `nemo-l2-current-raw-chat-stream.sse.gz`
- `nemo-l2-current-raw-health.json`
- `nemo-l2-explicit-off-stream.json`
- `nemo-l2-explicit-off-first-paint.png`
- `nemo-l2-explicit-off-final.png`
- `nemo-l2-off-health-start.json`
- `nemo-l2-final-on-settings.png`
- `nemo-l2-final-on-health.json`
- `nemo-l2-omni-tests.xml`
- `nemo-l2-server-tests.xml`

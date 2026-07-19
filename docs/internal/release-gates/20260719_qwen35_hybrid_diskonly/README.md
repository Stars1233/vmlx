# Qwen3.6 JANGTQ hybrid block-SSD cache with Paged RAM Off

Date: 2026-07-19

Branch: `reconcile/1.5.68`

Base before this scoped change: `dab2f53b8`

Model: `dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`

Bundle: `/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK`

Session: `100cb088-bf3c-4728-863a-b56182b27882` on `127.0.0.1:8029`

## Verdict

`FIXED_SOURCE + PASS_LIVE_SCOPED` for the exercised hybrid Qwen/JANGTQ
disk-only row. Paged RAM can be explicitly Off while Block Disk L2 remains On.
The attention-KV portion restores q4 TurboQuant-native blocks from SSD and the
typed SSM/GDN companion state restores through its architecture-specific L2.
Exact, restart, and changed-suffix partial-prefix requests all reused cache and
kept the paged-KV resident-byte budget at zero.

This is not a release verdict. Paged-On RAM-to-SSD tier ordering, explicit TQ
Off, Ollama/Anthropic parity, signed packaging, and other model families remain
separate open rows.

## Source trace

- `panel/src/shared/cacheControlPolicy.ts` adds the explicit
  `architectureSupportsBlockDiskOnly` policy input. It permits Paged Off only
  for a supported hybrid architecture with Block Disk L2 On; it does not relax
  native/typed exception families.
- `panel/src/main/sessions.ts`, `SessionConfigForm.tsx`, and
  `SessionSettings.tsx` use the same effective policy for migration, preview,
  persisted settings, and launch argv. Disk-only preview omits RAM cache-limit
  arguments.
- `vmlx_engine/scheduler.py` and `vmlx_engine/mllm_scheduler.py` initialize the
  block-aware backend when either Paged RAM or Block Disk L2 is enabled. The
  disk-only manager has `max_resident_bytes=0`, refuses a silent RAM fallback
  when SSD initialization fails, and retains the typed hybrid companion route.
- `vmlx_engine/mllm_batch_generator.py` and `scheduler.py` report
  `block-disk+ssm+tq-native` only after the request actually reconstructs
  TurboQuant-native blocks. They no longer label this route `paged`.
- Tests cover UI policy, preview/argv parity, disk-only initialization, hybrid
  companion restore, telemetry detail, and quantization-scoped disk keys.

## Bundle and launch truth

The real bundle health identifies `weight_format=mxtq` / JANGTQ. This is the
Hadamard/codebook JANGTQ route, not affine JANG and not base MLX MXFP.

The real Electron Session Settings UI saved:

```json
{
  "usePagedCache": false,
  "enableBlockDiskCache": true,
  "blockDiskCacheDir": "/Users/eric/.cache/vmlx-engine/live-proof-q35-hybrid-diskonly-20260719",
  "kvCacheQuantization": "auto",
  "reasoningParser": "qwen3",
  "toolCallParser": "qwen",
  "enableAutoToolChoice": true
}
```

The live process used:

```text
... vmlx_engine.cli serve ... --port 8029 --is-mllm --continuous-batching
--tool-call-parser qwen --enable-auto-tool-choice --reasoning-parser qwen3
--no-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000
--enable-block-disk-cache
--block-disk-cache-dir /Users/eric/.cache/vmlx-engine/live-proof-q35-hybrid-diskonly-20260719
--block-disk-cache-max-gb 10 --stream-interval 1
```

The full Electron-main relaunch log contains:

```text
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
```

## Live Electron cache matrix

| Row | Prompt | Reused | Detail | TTFT | Visible result |
|---|---:|---:|---|---:|---|
| Cold 569 | 5,166 | 0 | cold | 4.98s | exact `Q35-HYBRID-DISK-COLD-DONE` |
| Same-process 572, pre-fix | 5,166 | 5,165 | incorrectly `paged+ssm+disk` | 0.63s | exact marker; retained as live red |
| Stop/Start 575 | 5,166 | 5,165 | `block-disk+ssm+tq-native` | 0.45s | exact marker |
| Changed suffix 578 | 5,175 | 5,120 | `block-disk+ssm+tq-native` | 0.67s | exact `Q35-HYBRID-DISK-PARTIAL-DONE` |

After the partial request, health reported:

- `backend_mode=block_disk_only`, `paged_ram_enabled=false`, `disk_only=true`
- 81 blocks / 5,165 attention-KV tokens on SSD
- 321 actual SSD block hits and 321 TurboQuant-native hits
- zero paged-KV resident bytes and a zero paged-KV resident-byte budget
- typed SSM companion L2 enabled with 15,459 tokens on SSD and two disk hits

The SSM companion is architecture-specific state and may exist transiently or
in its separate bounded companion memory cache. The zero-RAM claim here is
strictly the Paged KV payload/budget; this gate does not claim the entire model
process has no cache-related memory.

## Streaming and tool loop

- Raw Chat SSE: 256 separate reasoning deltas, 169 content deltas, terminal
  `stop`, usage events, and `[DONE]` in `q35-raw-chat-stream.sse.gz`.
- Raw Responses SSE: 256 reasoning-summary deltas, 151 output-text deltas, one
  output-text done, one output-item done, and one `response.completed` in
  `q35-raw-responses-stream.sse.gz`.
- Electron row 581 made exactly one real `file_info(panel/package.json)` call,
  received `Size: 5.2 KB`, and exact-finaled
  `Q35-JT-DISK-TOOL-DONE SIZE=5.2 KB` with separate reasoning and no warning.

The two raw calculation prompts included extra visible math prose before their
requested lines. Streaming/terminal behavior passed; strict answer formatting
is therefore `PARTIAL`, not hidden as a pass.

## Validation

- Python expanded cache/scheduler selection: `911 passed, 5 deselected`.
- Panel cache/settings selection: `304 passed`.
- Panel TypeScript typecheck: passed.
- `git diff --check`: required before the scoped commit.

## Evidence files

- UI: `q35-settings-paged-off-block-l2.png`,
  `q35-cli-preview-diskonly.png`, `q35-live-loaded-card.png`,
  `q35-cold-complete.png`, `q35-restart-restore-complete.png`,
  `q35-partial-complete.png`, `q35-tool-loop-complete.png`, and
  `q35-tool-loop-logs.png`.
- Pre-fix live red: `q35-warm-complete-telemetry-red.png`.
- Health snapshots: `q35-health-*.json`.
- Raw APIs: `q35-raw-chat-stream.sse.gz` and
  `q35-raw-responses-stream.sse.gz`.
- Electron main: `vmlx-electron-q35-hybrid-diskonly-20260719.log.gz`.

## Retained open rows

- Paged On: prove resident-RAM lookup first, SSD refault after RAM eviction,
  partial block reuse at both tiers, and full prefill only when neither tier
  matches.
- Explicit TurboQuant Off for this hybrid artifact.
- Ollama and Anthropic stream/non-stream/tool continuation for this gate.
- Fault-injected SSD initialization/write failure and recovery.
- Remaining model families, media-salt rows, full suites/build, bundled-Python
  refresh, signing/notarization, install smoke, and publication.

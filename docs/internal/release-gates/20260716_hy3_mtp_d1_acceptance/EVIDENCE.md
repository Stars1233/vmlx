# HY3 native MTP D1 live acceptance and speed gate

Date: 2026-07-16

Model: `jangq-ai/Hy3-JANG_2K-MTP`

Live app: Electron development panel attached over CDP at `127.0.0.1:9335`; model endpoint `127.0.0.1:8010`.

## Verdict

- PASS: Electron Native MTP Off and Deterministic D1 controls both applied through `Save & Restart`.
- PASS: process health reported `runtime_disabled` for Off and `runtime_active=true`, `effective_depth=1` for D1.
- PASS: all six controlled greedy responses exactly matched integers 1 through 100, with 200 completion tokens and `finish_reason=stop`.
- PASS: D1 warm median wall time was 16.081931 s versus 21.234247 s Off, a 24.264% reduction (12.44 versus 9.42 completion tok/s from wall time).
- PASS: D1 telemetry was backed by real verify/draft execution. Across the controlled D1 arm: 414 cycles, 414 drafted tokens, 180 accepted tokens, 43.478% acceptance.
- PASS: Electron Perf visibly rendered D1, 60/138 last-request acceptance, depth rate, verify/MTP forward counts, and native cache/TQ state.
- PASS: Electron chat row 1944 called `file_info(panel/package.json)` exactly once and ended exactly `HY3-D1-TOOL1-DONE`, with no reasoning leak.
- PASS: post-tool health recorded D1 execution plus a reconstructed 43-token paged prefix; block L2 totals reached 69 native TQ writes and 7 native TQ hits.
- PARTIAL: phase timing labels are not isolated unless `VMLINUX_MTP_PROFILE` is enabled. Health explicitly reports `profiled_phase_timing=false`; wall-time A/B is authoritative here.
- PARTIAL: the model sidecar's historical near-full acceptance claim was not reproduced. Current controlled acceptance was 43.478%, while D1 still produced a measurable speedup.

## Controlled wall-time samples

| Mode | Run 1 | Run 2 | Run 3 | Median | Exact |
|---|---:|---:|---:|---:|---|
| Off | 23.496679 s | 17.986732 s | 21.234247 s | 21.234247 s | 3/3 |
| D1 deterministic | 22.627244 s | 16.006324 s | 16.081931 s | 16.081931 s | 3/3 |

Request controls for both arms: `temperature=0`, `top_p=1`, `max_tokens=512`, `enable_thinking=false`, identical prompt and process-restart/JIT position.

## Source trace

- `vmlx_engine/patches/mlx_lm_mtp/batch_generator.py`: publishes locked process-local request/totals snapshots and counts depth acceptance plus seed/verify/MTP forwards.
- `vmlx_engine/scheduler.py`: adds the finished BatchGenerator MTP snapshot to scheduler health stats.
- `tests/test_mtp_telemetry_edge_cases.py`: pins acceptance, per-depth rates, forwards, and aggregate publication.

## Tests

- 177/177 native-MTP-focused tests passed.
- 54/54 selected `test_engine_audit.py` MTP/health/scheduler tests passed.
- Python compile and `git diff --check` passed.

## Open issues discovered, not hidden by this gate

- Chat Settings Min P displays `0 = disabled`, but setting zero reset/persisted as null/0.05 during the controlled setup. This remains open for the settings-parity pass.
- Perf displayed `Attention KV L2 disabled` while health showed native TQ block writes/hits. The label/semantic mapping remains open for the cache-settings parity pass.
- Per-layer TurboQuant INFO telemetry can evict the MTP completion line from the bounded in-app Logs buffer. Health telemetry now preserves the counters, but log rate limiting remains open.

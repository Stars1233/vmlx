# openPangu long-context typed-snapshot admission guard

Status: `SOURCE+TEST+ELECTRON+API PASS` for the pre-copy admission guard;
`PARTIAL` for very-long native-cache reuse and tight-budget two-pass latency.

## Artifact and route

- Bundle: `/Volumes/EricsLLMDrive/jangq-ai/openPangu-2.0-Flash-JANG_3M`
- Family: `openpangu_v2`
- Quantization: affine importance profile `JANG_3M` (3/4/8-bit mix), not
  JANGTQ/MXTQ and not base MLX MXFP.
- Cache: native path-dependent `OpenPanguV2LayerCache` only. Generic
  TurboQuant, generic paged blocks, and block L2 remain disabled because the
  exact boundary includes MLA KV, DSA indexer state, rotating-SWA metadata,
  and three causal-convolution states per layer.
- Electron endpoint: `127.0.0.1:8027`, real UI Start button, CDP 9335.

## Reproduction and owner

The same 145,480-character deterministic archive was sent from a fresh
Electron chat before and after the patch. The UI prompt tokenized to 43,980
tokens and requested exact recall of three anchors.

Before the guard, `SingleBatchGenerator` always deep-copied the native N-1
cache boundary before the scheduler checked backend capacity. The copy was
22,090.8 MB, larger than both the 9,656 MB effective RAM entry cap and the
10 GB prompt-disk cap. It was eventually rejected, but the copy had already
inflated TTFT and peak Metal memory.

After the guard, the generator estimates the live typed boundary without
materializing a copy and skips snapshot creation when no enabled backend can
retain it. The model still performs its required full prefill and returns the
same exact answer; it merely avoids constructing a cache entry that must be
discarded.

## Source trace

- `vmlx_engine/utils/single_batch_generator.py` accepts
  `prompt_snapshot_max_bytes`, estimates the typed boundary before clone,
  records estimate/skip telemetry, and uses the guarded clone for typed paths.
- `vmlx_engine/scheduler.py` derives the ceiling from enabled RAM/disk
  backends, passes it into the generator, and exposes the values through
  scheduler and cache telemetry.
- `tests/test_openpangu_v2.py` proves an oversize boundary is rejected before
  the deep-copy function can run.

Focused current-source regression set: 124 passed across openPangu,
single-active generation, memory cache, disk cache, cache architecture,
isolation, and terminal-cleanup tests.

## Live Electron A/B

| Metric | Before | After |
|---|---:|---:|
| Prompt tokens | 43,980 | 43,980 |
| Exact visible answer | yes | yes |
| Reasoning/content separated | yes | yes |
| TTFT | 186.35s | 103.20s |
| Prompt processing | 236.0 tok/s | 426.2 tok/s |
| Total generation time | 199.6s | 113.6s |
| Peak Metal | 138,814.3 MB | 115,551.8 MB |
| Oversize snapshot copied | yes, then rejected | no |
| RAM or L2 entry falsely claimed | no | no |

The post-patch UI screenshot shows the exact answer
`cedar-7319|quartz-4821|harbor-9652`, a separate Reasoning rail, and the
43,980-token/103.20s metrics.

## Raw Responses streaming proof

A direct `/v1/responses` replay used the same archive, temperature 0, and a
deliberately tight `max_output_tokens=256`:

- SSE opened immediately with lifecycle events.
- First reasoning delta: 96.912688s.
- 256 progressive `response.reasoning_summary_text.delta` events.
- The tight cap ended the first pass inside reasoning, so the existing bounded
  tools-free answer pass performed a second full prefill.
- First visible content delta: 189.709594s.
- 23 progressive `response.output_text.delta` events.
- Exact answer, `response.output_text.done`, and `response.completed` at
  191.089278s.
- Health during the request reported a 20,885,864,448-byte estimate, a
  10,737,418,240-byte limit, and two skips (one for each full-prefill pass).

This proves the reasoning/content/terminal wire contract is streaming; the
long pause is prefill latency, not terminal batching. It also retains a real
performance limitation: a tight output cap can force a second full prefill
when the first pass consumes its budget in reasoning and the native prompt
boundary is too large to cache.

## Retained limitations

- A boundary larger than every configured backend is intentionally not cached;
  an identical repeat must full-prefill. No generic TQ substitution is allowed
  for this native composite.
- The tight-budget visible-answer fallback streams correctly but can double
  long-prompt prefill cost. This remains `PARTIAL`; no hidden sampler clamp,
  prompt coercion, or fabricated answer was introduced.
- The advertised 524,288-token context limit is not live-proven. This row
  proves exact retrieval at about 44k UI tokens and bounds the oversize-cache
  behavior.

## Evidence files

- `pangu-long-no-delta-fail.png` — pre-fix UI waiting during the 186s TTFT.
- `pangu-long-guard-retry-final.png` — post-fix Electron exact answer/metrics.
- `pangu-long-electron-rows.json` — pre/post DB rows and metrics.
- `pangu-long-filtered-logs.txt` — model/cache/guard/fallback timeline.
- `pangu-long-health-after.json` and `pangu-long-cache-stats-after.json` —
  current telemetry.
- `pangu-long-api-sse.jsonl` and `pangu-long-api-summary.json` — timed raw
  Responses events and terminal object.
- `pangu-long-prompt.txt` — deterministic prompt used for both UI and API.

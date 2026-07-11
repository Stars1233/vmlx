# FINAL — reasoning-on and Ollama streaming re-verification

Date: 2026-07-11

## Verdict

**SHIP** for the reasoning/streaming change set tested here.

This is a scoped verdict for Hy3 reasoning-on across Chat Completions,
Responses, Anthropic Messages, and Ollama Chat. It is not a claim that the
broader vMLX/MLXStudio release matrix is release-ready.

## Provenance

- Source baseline: `f2b7e8c12` (`bounded answer-pass floor`).
- Model: `/Volumes/EricsLLMDrive/jangq-ai/Hy3-JANG_2K-MTP`.
- Served name: `jangq-ai/Hy3-JANG_2K-MTP`.
- Standalone engine: `127.0.0.1:8010`; stopped after proof.
- Runtime: `family=hy_v3`, `qwen3` reasoning parser, `hunyuan` tool parser,
  native MTP depth 1, TurboQuant KV objects with stored-prefix q4.
- Neither the dev app, CDP, nor the production app was touched.

## Final reasoning-on matrix

Each usage cell is `turn1 / turn2 / turn3`; parenthesized values are overage
above the requested 384-token cap. Stream cells also show content-delta counts.

| Route | Mode | Usage | Content deltas | Grade |
|---|---|---:|---:|---|
| Chat Completions | non-stream | 228 / 396 (+12) / 420 (+36) | — | PASS |
| Chat Completions | stream | 393 (+9) / 300 / 420 (+36) | 8 / 5 / 31 | PASS |
| Responses | non-stream | 393 (+9) / 300 / 420 (+36) | — | PASS |
| Responses | stream | 393 (+9) / 300 / 420 (+36) | 8 / 11 / 31 | PASS |
| Anthropic Messages | non-stream | 393 (+9) / 300 / 420 (+36) | — | PASS |
| Anthropic Messages | stream | 393 (+9) / 300 / 420 (+36) | 8 / 5 / 31 | PASS |
| Ollama Chat | non-stream | 393 (+9) / 300 / 420 (+36) | — | PASS |
| Ollama Chat | stream | 393 (+9) / 300 / 420 (+36) | 8 / 5 / 31 | PASS |

All 8 route/mode sequences and all 24 turns passed. Every turn had non-empty,
coherent visible content, separated reasoning, no raw reasoning/tool-tag leak,
no repeated 8-gram loop, and usage overage no greater than the 48-token floor.
Every turn 3 recalled `TURN1`, `TURN2`, and `ORBIT-731` and ended with
`FINAL-CHECK`.

## Ollama stream fix and live wire proof

The misroute had four interacting causes:

1. `hy_v3` was not armed for the bounded reasoning answer pass.
2. A copied retry retained `reasoning_effort=high`, so Hy3 stayed on its
   thinking rail even after `enable_thinking=False`.
3. A length-capped first pass could emit a partial visible prefix and suppress
   the direct answer retry.
4. Ollama emitted the provisional first-pass `done:true` before answer-pass
   content and dropped the later terminal/usage line.

The fix arms Hy3, forces both engine and template retry kwargs to
`reasoning_effort=no_think`, replaces length-truncated prefixes with the bounded
direct answer, streams finalized short Hy3 content in multiple chunks, and
defers/merges Ollama terminal chunks into one final line.

Final Ollama stream event ordering:

| Turn | Content event range | Sole `done:true` | Last event | Result |
|---|---:|---:|---:|---|
| 1 | 384–391 | 392 | 392 | PASS |
| 2 | 287–291 | 292 | 292 | PASS |
| 3 | 384–414 | 415 | 415 | PASS |

Thus visible answers are incremental `message.content`; `message.thinking`
contains only the reasoning rail; and the sole terminal event follows all
content.

## Warm greedy determinism

Seeded temperature-zero samplers were request-local but native MTP inferred
"stochastic" from sampler-object presence. That sent argmax through speculative
acceptance and caused the optional trailing period. The sampler now marks the
actual greedy contract and MTP uses identity verification.

| Route | Warm run 1 | Warm run 2 | Grade |
|---|---|---|---|
| Chat Completions | `DET-731` (4) | `DET-731` (4) | PASS |
| Responses | `DET-731` (4) | `DET-731` (4) | PASS |
| Anthropic Messages | `DET-731` (4) | `DET-731` (4) | PASS |
| Ollama Chat | `DET-731` (4) | `DET-731` (4) | PASS |

## Tests and A/B

- Focused final set: 248 passed.
- Full baseline: 5,939 tests; 5,792 passed; 53 failed; 94 skipped; 92 deselected.
- Full post-change: 5,945 tests; 5,798 passed; 53 failed; 94 skipped; 92 deselected.
- Exact JUnit failure-ID comparison: 0 new, 0 resolved, identical 53-test
  pre-existing failure set.
- Python compile and `git diff --check`: passed.

## Artifacts

- `all-routes-final.json` — final live wire capture, `status=pass`, failures `[]`.
- `all-routes-final-failures.json` — empty failure bundle.
- `full-suite-baseline.xml` and `full-suite-post-final.xml` — exact A/B evidence.
- `hy3-server-all-routes-final.log` — final startup/runtime/server log.
- `run_reverify.py` — bounded-floor and terminal-order grader.

SHA-256:

```text
331dd10dad221d7552b08c613f2884120f841a36ff9d88eb5cbf179200d1c74a  all-routes-final.json
02491dc1a1d0a89ea1d39d6ec576ff53ef76d4d5367d620b1b36dcedd74c86fb  all-routes-final-failures.json
7ce84e0b15bf28745c1009c42f606e482daac2d6e23bdc717b542133adb7098f  full-suite-baseline.xml
9506e2a92e573a15811b2ef719d33f3e1fd409d0e49dbea4dafe792ea4e3bce5  full-suite-post-final.xml
```

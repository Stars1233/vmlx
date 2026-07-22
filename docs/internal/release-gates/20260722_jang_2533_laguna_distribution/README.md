# JANG 2.5.33 and Laguna mixed-affine distribution repair

Date: 2026-07-22 (America/Los_Angeles)

Verdict: `VERIFIED-LIVE_SCOPED`; the v1.6.16 campaign remains
`PARTIAL / NOT RELEASE-READY`.

## Defect and exact boundary

The Laguna error below is not a 576-wide `g_proj` slice:

```text
[dequantize] matrix (...,576) vs scales (...,48), group_size=64, bits=8
```

For the inspected S-2.1 JANG bundles, `576 = 3072 * 6 / 32` is the packed
width of a 3072-input 6-bit affine tensor and `48 = 3072 / 64` is its scale
group count. The bundle's attention `g_proj` is 8-bit. The failure occurs when
a runtime applies the top-level 8-bit setting to a mixed-bit 6-bit module.

The signed v1.6.15 Sequoia and Tahoe apps already contain a capable bundled
runtime, so their retained signed-app generation proofs remain valid. The
public Python package surface was different: PyPI `vmlx==1.6.15` declared only
`jang>=2.5.29`, and that floor did not guarantee the per-module mixed-affine
runtime contract. Thus Python/CLI v1.6.15 could legally resolve a stale JANG
wheel and reproduce the defect.

## Source and distribution repair

JANG commit `b788273e47b0f5be834d85c5b31e757b34e4d950` is published as
`jang==2.5.33` and tag/release `v2.5.33`. It adds:

- `LAGUNA_MIXED_AFFINE_RUNTIME_VERSION = 1`;
- shape-derived affine bit inference for each Laguna module;
- an exact `(576, 48, group_size=64) -> 6 bits` regression.

Published artifact digests:

- wheel: `ada2af21562662842fb95125dfa46e26b7c0a7ced498e1b9c2fd3262cf51b0f4`;
- sdist: `9ca5733211b3f78d9a30d7be42458c9d33c573f72a20e51696eaa94b64020385`.

vMLX commit `b6d38eac7` raises the dependency floor to `jang>=2.5.33` and
rejects mixed-affine Laguna bundles before model execution when the imported
runtime lacks the marker. Non-affine Laguna JANGTQ/MXTQ and MXFP4 formats do
not enter this affine marker gate. Commit `e4c6762ce` logs the imported runtime
module, marker, and mixed-affine classification at model load.

## Verification

Source/test proof:

- JANG focused Laguna tests: 16 passed;
- JANG full suite on both clean source boxes: 574 passed, 37 skipped;
- vMLX exact-head Laguna/reasoning/config focused set on both boxes: 370
  passed;
- panel engine path/isolation tests on the current source: 7 passed;
- fresh no-cache PyPI wheel import reported version 2.5.33, runtime marker 1,
  runtime SHA-256 `96091e33...`, and inferred 6 bits for `(576,48)`.

Current-source live Electron proof on `erics-m5-max.local`:

- the real Sessions **Start** control loaded
  `jangq-ai/Laguna-S-2.1-JANG_2L` as PID 39057 on port 8018;
- the child environment had
  `PYTHONPATH=/Users/eric/mlx/vllm-mlx-release-1.6.13`, proving the shared venv
  launcher imported the synchronized release checkout;
- the session log recorded JANG runtime marker 1 from the physical 2.5.33
  site-packages module before `Laguna loaded`;
- live health identified affine JANG_2L, 48 layers, the typed
  `mixed_swa_kv_v1` cache, q4 storage only for `full_attention_kv`, native
  sliding-window/rotating metadata, Paged RAM On, and Block Disk L2 On;
- Chat Settings visibly matched the bundle defaults: temperature 1.0, top-p
  1.0, top-k 20, min-p 0, repetition penalty 1.0, and model-default max output
  32768;
- an explicit-thinking UI turn persisted 2,000 reasoning characters
  separately from non-empty visible content; the model duplicated the requested
  marker in visible prose, so strict marker-only behavior is not claimed;
- the next turn emitted exactly one real `file_info(panel/package.json)` call,
  accepted the real 5.2 KB result, persisted separate reasoning, and completed
  visibly with `R16-LAG-CURRENTSRC-TOOL-DONE` without warnings;
- that tool continuation restored 5,125 `paged+tq-native` prompt tokens.

Current-source raw API proof:

- Responses Auto emitted separate reasoning-summary and output-text events,
  one completed terminal, visible `YES`, and no inline marker leakage;
- repeating the same prompt produced different reasoning text and an 81-token
  `paged+tq-native` hit;
- explicit `enable_thinking=false` emitted zero reasoning and visible `YES`;
- Chat Completions emitted 935 reasoning characters through
  `delta.reasoning_content`, visible `YES` through `delta.content`, a normal
  stop, and no inline marker leakage.

## Retained artifacts

- `r16-laguna-2533-source-binding.json`
- `r16-laguna-2533-session-log-proof.json`
- `r16-laguna-2533-current-health.json`
- `r16-laguna-2533-current-ui-db.json`
- `r16_laguna_currentsrc_api.json`
- `r16-laguna-current-source-reason.png`
- `r16-laguna-current-source-tool.png`

This closes the package/runtime provenance defect and the named current-source
Laguna UI plus Chat/Responses regression only. It does not close Laguna's
Paged-Off restart, long eviction, long SWA quality, four-protocol agentic,
gateway, media, full-suite, packaging, notarization, or release rows.

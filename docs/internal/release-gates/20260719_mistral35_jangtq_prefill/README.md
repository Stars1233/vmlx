# Mistral Medium 3.5 JANGTQ prefill/output gate (2026-07-19)

## Verdict

`BLOCKED_CURRENT_ARTIFACT_RUNTIME` for release use. The official JANGTQ bundle now passes strict loader hydration, but neither available compute path produced a coherent visible answer in the live Electron app:

- legacy TurboQuant path: prefill did not reach TTFT after more than two minutes;
- MPP NAX `auto`: prefill reached TTFT, but decode emitted newline-only tokens;
- dtype-aware FP32 NAX rerun: reproduced the same newline-only output.

The broad dense-Mistral `auto` exception introduced by `34ec5189d` is therefore reverted at the current head. This gate does **not** blame or mutate the official quantized artifact. The remaining failure may be model-port, prompt/logit, or dense JANGTQ2 integration behavior and needs a separate matched reference/runtime comparison.

## Artifact-grounded configuration

Bundle: `/Volumes/EricsLLMDrive/jangq-ai/Mistral-Medium-3.5-128B-JANGTQ`

- outer `model_type=mistral3`; inner `text_config.model_type=ministral3`
- `dtype=bfloat16`; 88 dense decoder layers
- hidden/intermediate sizes `12288/28672`
- attention heads/KV heads/head dim `96/8/128`
- `weight_format=mxtq`, profile `JANGTQ2`
- text decoder `mxtq_bits=2`; embeddings 8-bit; lm_head/vision/projector passthrough FP16
- vision configuration and weights are present; this gate exercised text only
- no MTP tensors

Live `/health` recorded 616 TurboQuant target tensors and `swapped=616 skipped(no-module)=0`. See `health-auto-fail.json` and the Logs screenshot.

## Source trace

1. `vmlx_engine/utils/jang_loader.py` strict Mistral3 hydration was re-enabled by `e46ceb4a8`. This removed the old unconditional loader rejection but did not assert output correctness.
2. `vmlx_engine/cli.py::_apply_jangtq_mpp_nax_policy` normally disables the unverified MPP NAX auto lane for MXTQ/JANGTQ bundles.
3. `34ec5189d` temporarily exempted dense `mistral3/ministral3` after isolated projection checks. Live full-model output disproved that exception, so it is removed in the corrective commit associated with this evidence.
4. The separate Jang TurboQuant diagnostic changed BF16/FP32 NAX inputs to FP32 A/B accumulation and passed 23 focused Metal-kernel tests. It improved projection-level agreement but did not repair full-model newline output, so it is documented as kernel evidence only—not as a model closure.

## Live Electron evidence

All model launches and generations used the real dev Electron app over CDP 9335, session `48caa5b6-979a-4307-9329-2e340c081a6f`, endpoint `127.0.0.1:8001`.

### Legacy kernel

The first post-loader turn remained in prefill for more than two minutes without TTFT or SSE content. It was cancelled in the real UI.

- `legacy-prefill-stall.png`
- `legacy-prefill-cancelled.png`

### First MPP NAX auto attempt

The `34ec5189d` exception reduced prefill time, but the assistant emitted newline-only tokens with no visible content. This is a live output failure, not a pass.

- `first-auto-newline-loop.png`

### FP32 A/B diagnostic rerun

Chat settings were opened in Electron and Max Tokens was explicitly set to 64. The request log proves:

```text
wireApi=responses
stream=true
max_tokens=64
temperature=0.0
top_p=1.0
has_tools=false
```

Prompt:

```text
[M35JT-UI-FP32] Do not call tools. Reply exactly M35JT-FP32-DONE and nothing else.
```

Observed incremental DB content lengths before cancellation:

```text
8 -> 16 -> 22 newline characters; reasoning length 0; visible marker absent
```

The UI showed `27 tokens`, `0.7 t/s`, `398 prompt`, `14.29s TTFT`, and no visible answer. After 68.7 seconds and 39 tokens the real UI Stop-generating control was used; the persisted assistant row became `[Generation interrupted]`.

- `m35jt-fp32-newline-fail.png`
- `m35jt-fp32-fail-logs.png`
- `health-auto-fail.json`

## Focused verification

- vMLX corrective policy tests: `12 passed` for `TestJangTqMppNaxCliPolicy`
- Jang MPP NAX tests after dtype diagnostic: `23 passed`
- isolated official-weight projection checks covered q/k/v/o/gate/up/down at model shapes with cosine approximately `0.999998`; those checks did not predict the full-model failure and are not treated as runtime proof

## Missing proof / next work

- coherent non-empty visible answer: **FAIL**
- progressive content SSE with terminal completion: **FAIL**
- reasoning separation: **UNTESTABLE until coherent generation**
- tool call and post-tool continuation: **UNTESTABLE**
- warm/paged/L2/TQ cache correctness: **UNTESTABLE for output correctness**
- multimodal/Pixtral path: **UNTESTED**
- matched same-artifact comparison against a known-good JANG reference runtime: **REQUIRED**

This model row remains blocked and must not be included in a release-ready matrix until the full-model integration failure is understood and re-proven live.

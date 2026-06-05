# Cross-Model Runtime Failure Classes: Modality Metadata, Tool Dialects, Loop Control

Date: 2026-06-05

Primary concrete repro: `dealignai/Step-3.7-Flash-JANG_2L-CRACK` on vMLX 1.5.55.

Scope: this is not a Step-only issue. Treat the Step Flash CRACK result as one concrete instance of a broader release-gate class that must be reproduced across model architecture, runtime, quantization, config, and modality combinations before claiming a model family is production-cleared.

## Executive Summary

Step-3.7 Flash JANG_2L CRACK is stable as a text-only model when served with model-card-aligned metadata (`jang_config.architecture.has_vision=false`), but default downloaded metadata can route it into an unsupported Step3p7 VLM path and kill the server during generation. The same class of issue can affect any model where artifact metadata advertises capabilities that the current runtime path cannot safely execute.

The runtime issue is separable from behavior quality:

- Runtime stability: text-only path is stable after metadata correction.
- Unsupported modality routing: VLM auto-detection can select an unsafe path.
- Tool discipline: model may emit raw XML-like tool text instead of API `tool_calls`.
- Agent loop control: model may repeat tool calls after mock/tool outputs instead of synthesizing a final answer.
- Thinking/template mismatch: templates may inject thinking markers even when runtime asks for thinking off.

These must be tested and tracked per family, not inferred from one passing model.

## Concrete Step Flash CRACK Evidence

Model: `dealignai/Step-3.7-Flash-JANG_2L-CRACK`

Runtime tested: vMLX 1.5.55

Machine: Apple Silicon arm64, 128 GB unified memory

Local path observed by reviewer:

```text
/Users/vera/models/Step-3.7-Flash-JANG_2L-CRACK
```

Observed metadata:

```text
config.json:
  model_type: step3p7
  architectures: Step3p7ForConditionalGeneration
  vision_config: present
  audio_config: absent
  max_position_embeddings: 262144

jang_config.json:
  architecture.has_audio: false
  architecture.has_mtp_tensors: false
  architecture.has_vision: true
  architecture.text_model_type: step3p5
  architecture.type: step3p7
  capabilities.supports_thinking: true
  capabilities.supports_tools: true
  capabilities.think_in_template: true
  capabilities.tool_parser: step3p5
```

Default metadata routed vMLX into MLLM/VLM:

```text
is_mllm_model(...Step-3.7-Flash-JANG_2L-CRACK): tier=jang_config_explicit_true result=True
SimpleEngine loaded: ... (MLLM=True)
```

Failure symptoms in unsupported MLLM path:

```text
RemoteProtocolError('Server disconnected without sending a response.')
ConnectError('[Errno 61] Connection refused')
```

Final log lines before crash on a normal safe prompt:

```text
Resolved sampling kwargs ... {'temperature': 0.2, 'top_p': 0.9, 'max_tokens': 256, 'enable_thinking': False}
MLLM.chat() called with 1 messages
Applying chat template with 1 messages, 0 images, 0 audio
Chat msg 0: role=user, content=Explain how to calm down when angry, avoid conflict, and leave safely. Give 5 co...
```

Important warning:

```text
Template for models/Step-3.7-Flash-JANG_2L-CRACK always injects <think> (ignores enable_thinking=False)
```

Text-only metadata view:

```text
/Users/vera/models/Step-3.7-Flash-JANG_2L-CRACK-textonly-view
```

Only `jang_config.json` was changed:

```json
{
  "architecture": {
    "has_vision": false
  }
}
```

Then vMLX classified it as text-only:

```text
is_mllm_model(...Step-3.7-Flash-JANG_2L-CRACK-textonly-view): tier=jang_config_explicit_false result=False
SimpleEngine loaded: ... (MLLM=False)
LLM model loaded (simple mode)
```

Working launch command:

```bash
vmlx serve /Users/vera/models/Step-3.7-Flash-JANG_2L-CRACK-textonly-view \
  --port 8093 --host 127.0.0.1 \
  --served-model-name step37-2l-crack-text \
  --max-tokens 512 --max-num-seqs 1 \
  --no-continuous-batching --disable-prefix-cache \
  --kv-cache-quantization none --disable-native-mtp \
  --tool-call-parser step3p5 --reasoning-parser qwen3 \
  --default-enable-thinking false --log-level INFO
```

Startup highlights:

```text
JANG v2 detected -- loading via mmap (instant)
Step-3.7 model family detected; normalizing text-runtime dispatch to step3p5.
JANG v2 loaded in 23.1s: Step-3.7-Flash-NVFP4 (3.4-bit avg)
Metal GPU memory after load: 77.92GB active, 77.92GB peak
System memory after load: 26.9GB available (79.0% used)
```

Text-only smoke after metadata correction:

- exact short text response: pass
- JSON object response: pass
- OpenAI tool call: pass
- longer safe text generation: pass
- unsafe/looping probes without server crash: pass
- post-probe health check: live

Fast eval:

```text
Score:              113.0/180.0
Percentage:         62.78%
Grade:              C
Tasks:              18/18 ok, 0 failed
eval tok/s:         16.5
latency p50/p95/p99 7.1s / 34.3s / 37.0s
budget compliance:  18/18, 100%
production_fit:     0.628
```

Result file from reviewer:

```text
/Users/vera/vera-eval/results/step37-2l-crack-textonly-vmlx155-fast_1780688714.json
```

## Cross-Model Failure Classes To Reproduce

### 1. Unsupported modality metadata routes into unsafe runtime

Pattern:

- Artifact advertises `vision_config`, `audio_config`, `video_config`, or `jang_config.architecture.has_vision=true`.
- vMLX treats artifact as MLLM/VLM/omni.
- Runtime family does not actually support that modality path.
- Server may crash during generation, often without a Python traceback.

Concrete Step instance:

- Step3p7 VLM is not implemented in current vMLX path.
- Metadata advertises vision.
- MLLM route can kill server mid-request.
- Text-only metadata view is stable.

Must reproduce across:

- Step3p7 text-only vs VLM metadata.
- Gemma 4 unified text/VL/audio/video configs and quants.
- Qwen VLM and Qwen MTP VLM paths.
- LFM text and any VL-advertised variants.
- Nemotron Omni audio/image/video paths.
- DSV4 dense/MoE text paths with any advertised media sidecars.
- JANG/JANGTQ/MXFP bundles with stale or overbroad metadata.

Required guard:

- If metadata advertises modality but runtime support is missing or known unsafe, fail closed or force an explicit `--allow-unsupported-modality-runtime` style override.
- Log exact source of modality detection: `config.json`, `jang_config.json`, sidecar, CLI override, model registry.
- Include family-specific message with the safe config workaround when known.

### 2. Tool-call dialect ambiguity

Pattern:

- Model receives OpenAI-style tools.
- Model sometimes emits proper API `tool_calls`.
- Under harder prompts it emits literal XML-ish text such as `<tool_call><function=...>` instead of structured `tool_calls`.
- Harness/client does not execute the tool because the output is visible text, not API structure.

Concrete Step instances:

- Some tasks produced correct OpenAI tool calls.
- Other tasks emitted literal XML-ish tool text for `write_file` or `terminal` instead of structured calls.

Must reproduce across:

- Step3p5/Step3p7 parser paths.
- Qwen tool parser paths.
- Gemma 3/Gemma 4 parser paths.
- DSV4 DSML parser paths.
- Zaya XML parser paths.
- MiniMax/Nemotron/Hunyuan/Kimi parser paths.
- Ollama gateway and OpenAI gateway paths.
- Streaming and non-streaming paths.

Required guard:

- If tools are supplied and parser detects raw known tool dialect in visible text, either convert to structured `tool_calls` when unambiguous or flag/log `raw_tool_dialect_leak`.
- Add release tests that distinguish correct tool call emission from visible raw tool text.

### 3. Agent loop control after tool output

Pattern:

- Model initiates reasonable diagnostics.
- After receiving mock/tool output, it repeats identical or near-identical commands.
- It fails to synthesize a final answer and burns the turn/tool budget.

Concrete Step instances:

- Repeated `top`, `id`, `systemctl`, `journalctl`, SSH, package install, or service-file reads.
- Several tasks exhausted 10 turns instead of finalizing.

Must reproduce across:

- Text-only dense models.
- JANG/JANGTQ models.
- MXFP models.
- MTP-enabled models.
- Models with native tool parsers vs fallback tool injection.
- Gateway paths: OpenAI Chat, Responses, Anthropic adapter, Ollama adapter.

Required guard:

- Add eval rows for repeated-command loops and final-answer synthesis after mock output.
- Track `tool_repeat_count`, `same_tool_same_args_count`, `final_answer_after_tool_output`, and `turn_budget_exhausted`.
- Do not mark a model agent-ready from first tool-call success alone.

### 4. Thinking/template mismatch

Pattern:

- Runtime requests `enable_thinking=false`.
- Template still injects `<think>` or family-specific reasoning tags.
- Parser/display may hide some leakage, but behavior can drift and agent tasks may degrade.

Concrete Step instance:

```text
Template ... always injects <think> (ignores enable_thinking=False)
```

Must reproduce across:

- Step3p5/Step3p7.
- Qwen reasoning templates.
- Gemma 4 channel/thought templates.
- DSV4 thinking/direct rails.
- Zaya/MiMo XML thinking templates.
- GPT-OSS/MiniMax/Nemotron reasoning parsers.

Required guard:

- Template render tests must inspect prompt text and final output for thinking markers under thinking off.
- Live tests must verify visible answer quality, not only marker stripping.

### 5. Silent native crash / missing traceback

Pattern:

- Server exits or disconnects mid-request.
- Client sees connection refused/disconnected.
- Logs stop before Python exception handling.

Concrete Step instance:

- Unsupported MLLM route died after `MLLM.chat()` and chat-template application.

Must reproduce across:

- VLM image prefill.
- Video frame prefill.
- Audio/omni dispatch.
- MTP draft paths.
- TurboQuant/JANGTQ kernels.
- DSV4 composite cache paths.

Required guard:

- Add parent-process crash sentinel and last-request breadcrumb: model family, modality route, parser IDs, thinking mode, cache mode, prompt/media token estimate, and last Python stage before Metal/native forward.
- Regression tests should assert a controlled 4xx/5xx or warning for unsupported route, not process death.

## Required Cross-Family Reproduction Matrix

Do not infer production readiness across this matrix. Each row needs explicit evidence.

| Axis | Required rows |
|---|---|
| Architecture | dense LLM, MoE, hybrid SSM/attention, VLM, omni/audio, DSV4 composite, Step3p7 |
| Quant/runtime | fp/bf16, MXFP4, MXFP8, JANG_K, JANG_4M, JANG_2L, JANGTQ, TurboQuant KV, native MTP |
| API surface | OpenAI Chat, OpenAI Responses, legacy Completions, Anthropic, Ollama, panel gateway |
| Modality | text-only, image, video, audio, mixed media, metadata-advertised-but-runtime-unsupported |
| Cache | no cache, prefix cache, paged cache, L2 disk cache, MLLM media cache, DSV4 native cache, MTP with cache |
| Behavior | exact answer, JSON/schema, XML/tool call, multi-turn recall, tool after tool-output finalization, refusal/safety, long output |
| Streaming | non-streaming, streaming, streamed tool calls, streamed structured output validation |
| Lifecycle | load, health, request, recover after rejected media/error, soft sleep, JIT wake, unload/reload |

Minimum per-row evidence:

- Load classification with source of family/modality detection.
- One visible text smoke.
- One structured JSON/schema or XML/tool test if tools/structured output are supported.
- One multi-turn context test.
- One cache hit or explicit cache-disabled proof.
- One error/recovery test for unsupported modality or oversized media.
- No process death under the tested route.

## Current Recommended User Configuration For Step Flash CRACK

Until Step3p7 VLM is implemented and tested in vMLX, serve the model text-only:

```json
{
  "architecture": {
    "has_vision": false
  }
}
```

Recommended command:

```bash
vmlx serve /path/to/textonly-view \
  --served-model-name step37-2l-crack-text \
  --port 8093 \
  --max-tokens 512 \
  --max-num-seqs 1 \
  --no-continuous-batching \
  --disable-prefix-cache \
  --kv-cache-quantization none \
  --disable-native-mtp \
  --tool-call-parser step3p5 \
  --reasoning-parser qwen3 \
  --default-enable-thinking false
```

## Open Work

- Add vMLX guard for unsupported Step3p7 VLM routing and similar metadata/runtime mismatches.
- Add cross-family modality metadata audit to release gate.
- Add tool dialect leak detection/repair or explicit warnings.
- Add loop-control eval rows and metrics.
- Add thinking-template mismatch render/live tests per family.
- Add native crash breadcrumb/sentinel for unsupported VLM/audio/video/MTP/TurboQuant paths.

## Release Rule

A model family is not production-cleared from one successful text-only smoke or one successful tool call. Production clearance requires evidence across the matrix above, with unsupported metadata/runtime routes either working or failing closed with clear diagnostics.

## GitHub Tracking Issues

- Master cross-model matrix: https://github.com/jjang-ai/vmlx/issues/188
- Unsupported advertised modality/runtime guard: https://github.com/jjang-ai/vmlx/issues/189
- Raw tool dialect leaks, tool loops, thinking-template mismatch: https://github.com/jjang-ai/vmlx/issues/190
- Structured JSON/XML repair follow-up: https://github.com/jjang-ai/vmlx/issues/187

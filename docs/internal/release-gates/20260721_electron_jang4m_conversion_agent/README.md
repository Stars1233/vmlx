# Electron-created affine JANG conversion and agent loop — current source

Date: 2026-07-21

Status: `VERIFIED-LIVE_SCOPED` for the affine JANG text/conversion/agent loop;
`PARTIAL` for converted-model media and for general JANGTQ/MXTQ production.

Repo: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch: `codex/postrelease-ui-drawers-20260720`

Source cutoff: `f0b1617ea`

## Exact artifact identity

The real Electron converter started from the local BF16 source bundle:

`/Users/eric/models/OsaurusAgent-9b-BF16`

and produced:

`/Users/eric/models/Codex-Quant-Probe-OsaurusAgent-9b-JANG_4M`

This output is **affine JANG**, not JANGTQ/MXTQ and not base MLX MXFP:

- `model_type=qwen3_5`, architecture `Qwen3_5ForConditionalGeneration`
- profile `JANG_4M`, target 4 bits, measured 4.66 bits
- `quantization_backend=mx.quantize`, asymmetric affine groups of 64
- tensor bit widths 4/8 plus 16-bit passthrough tensors
- `hadamard_rotation=false`
- 250 quantized tensors and 72 passthrough tensors
- 10 present safetensor shards, 1,260 indexed tensors, no missing shard
- 6.23 GB reported weight size / 6.3 GB filesystem allocation

The converter's format selector does not claim that this generic Qwen path
creates JANGTQ/MXTQ. The distinct Zaya routing in
`vmlx_engine/commands/convert.py::_jang_convert_command` is a separate
JANGTQ-specific branch and was not exercised here.

## Active production source trace

The user-visible flow is owned by live production call sites:

1. `panel/src/renderer/src/components/tools/ModelConverter.tsx::runConvert`
   sends the selected profile/method/output to the developer IPC and renders
   streamed progress, completion, and `Serve Model`.
2. `panel/src/main/ipc/developer.ts::resolveCliSpawn` resolves the same
   installed engine as Sessions, strips inherited `PYTHONPATH`, and
   `developer:convert` builds `vmlx-engine convert ... --jang-profile ...`.
3. `vmlx_engine/commands/convert.py::convert_command` routes
   `--jang-profile` to `_jang_convert_command`, which calls the installed
   `jang_tools.convert.convert_model`, writes the output, and runs the
   post-conversion generation smoke test.
4. `ModelConverter`'s live `Serve Model` callback routes through the ordinary
   Create Session page; it does not spawn a hidden test server.
5. The generated model's real Electron chat used the ordinary Responses
   request builder/tool executor in `panel/src/main/ipc/chat.ts`.

The owning path and call sites were inspected for dead/test-only bypasses. No
model-specific output rewrite, prompt coercion, synthetic sampler default, or
compatibility branch was added, and no dead branch was found that was safe to
delete as part of this scoped proof.

## Real Electron conversion and eager load

The real dev Electron instance on CDP 9335 visibly selected:

- Tools -> Convert Model
- Format: `JANG`
- Profile: `4M — Standard`
- Quantization: `MSE-optimal`
- explicit output directory shown above

The Electron main process spawned exactly:

```text
/Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine convert \
  /Users/eric/models/OsaurusAgent-9b-BF16 \
  --jang-profile JANG_4M \
  --jang-method mse \
  --output /Users/eric/models/Codex-Quant-Probe-OsaurusAgent-9b-JANG_4M
```

The UI completed in 13.8 seconds and visibly reported:

```text
Conversion complete
DONE — JANG v2 (MLX-native)
Size: 6.23 GB
Avg bits: 4.66
PASS: Generated: 'The capital of France is Paris. This is a'
```

The real `Serve Model` button opened Create Session with the converted path.
The real `Launch Session` button created session
`ecba1b7b-c303-421d-b5f4-32c6b37da0ac` on port 8006 and PID 68166. Before any
chat request, health reported `model_loaded=true`, `last_request_time=null`,
6,388.1 MB active memory, and affine `mlx_affine_quantized_matmul` dispatch.
This is eager materialization evidence for this produced artifact.

The UI/DB/argv/health settings agreed on Qwen tool + reasoning parsers,
Auto/model-owned sampling (no synthetic generation overrides), continuous
batching, prefix cache, paged cache block size 64 / 1,000 blocks, Block L2 10
GB, and q4 TurboQuant storage only for the eight compatible attention-KV
layers. The other 24 hybrid SSM/GDN layers retain native companion state with
async rederive. This identity is recorded to prevent the affine weight codec
from being confused with the separate TurboQuant KV-storage codec.

## Electron reasoning, tool, and history proof

The first direct UI turn emitted 197 characters of separate reasoning, then
progressively painted and exact-finaled `QUANT-UI-T1-DONE` with non-empty
visible content.

The initial tool prompt intentionally exposed a test-setup error: SQLite and
Chat Settings both showed `builtin_tools_enabled=0`, and the request-shape log
showed `has_tools=false`. The model therefore returned raw JSON as ordinary
text. That was not a converter, quantized-weight, Qwen parser, or Responses
stream failure. After enabling **Built-in Coding Tools** through the visible
Chat Settings UI and keeping reasoning on **Auto**, a fresh chat inherited
`builtin_tools_enabled=1`; its request-shape log showed `has_tools=true`.

That corrected Electron turn:

- painted 12 distinct UI states in 1.484 seconds
- streamed 163 characters of reasoning separately
- emitted exactly one schema-valid `file_info` call
- executed the real working-directory-relative path `panel/package.json`
- persisted the actual result `Size: 5.2 KB`
- progressively painted the visible answer
- exact-finaled `QUANT-UI-T2-CORRECTED-DONE SIZE=5.2 KB`
- persisted one OpenAI tool call/result pair and no warning

The next same-chat turn explicitly prohibited another tool and used only the
prior result. It painted eight distinct states, made no tool call, kept 134
reasoning characters separate, and exact-finaled:

```text
QUANT-UI-T3-DONE PATH=panel/package.json SIZE=5.2 KB
```

Its 4,207-token prompt completed with 1.49 s TTFT, demonstrating that the
actual tool exchange was replayed into the next turn. This row does not infer
cache correctness from TTFT alone.

One early proof-harness screenshot caught the final UI while the answer text
still ended at `SIZE=5`, even though terminal metrics were already visible.
The persisted DB row already contained the full answer; one second later the
same live DOM and the stable screenshot showed the complete `SIZE=5.2 KB`.
Only the stable screenshot is retained as the visual final-state artifact.
The timed trace remains useful for progressive paint evidence but is not used
as the final-text oracle.

## Raw Responses SSE proof

`responses_tool_stream_probe.py` repeats the same logical agent loop directly
against the already Electron-loaded server; it does not start a second model.
`quant-responses-tool-stream-proof.json` records every timed SSE event.

The current run passed every check:

- round 1: 28 reasoning deltas, two function-argument deltas, exactly one
  `file_info({"path":"panel/package.json"})`, one completed terminal
- round 2: `previous_response_id` plus a real file-derived tool result, tools
  still present, no repeated tool call, 34 reasoning deltas, 15 progressive
  content deltas, exact terminal `QUANT-API-R2-GATE-DONE SIZE=5.2 KB`
- one `response.completed` per round
- round-2 usage: 256 cached input tokens,
  `cache_detail=paged+ssm+tq-native`
- elapsed: 837.823 ms then 789.351 ms

This closes the Responses streaming/tool-continuation surface for this
produced artifact. It does not substitute for the visual Electron evidence
above.

## Artifacts

- `quant-jang4m-ready.png`
- `quant-jang4m-complete-status.png`
- `quant-created-loaded.png`
- `quant-ui-t1.png`
- `quant-ui-t1-trace.json`
- `quant-ui-t2-corrected-stable.png`
- `quant-ui-t2-corrected-trace.json`
- `quant-ui-t3-stable.png`
- `quant-ui-t3-trace.json`
- `quant-responses-tool-stream-proof.json`
- `responses_tool_stream_probe.py`

All six retained screenshots were visually inspected.

## Honest boundary / next issue discovered

This is not a general quantizer certification. It proves one UI-created
Qwen3.5 affine JANG_4M bundle through conversion, eager load, coherent text,
Auto reasoning, Responses streaming, a real one-tool loop, and same-chat
history.

Still open:

- other affine profiles and custom mix
- force-overwrite and low-disk/unwritable-volume UX
- true interrupted-job resume (distinct from reconnecting to a live child)
- JANGTQ/MXTQ Hadamard-codebook conversion except its separate Zaya branch
- large MoE conversion, calibration/AWQ/imatrix modes
- Chat/Anthropic/Ollama for a newly converted bundle
- eviction, process-restart L2 restore, paged-off disk-only reuse, and partial
  block reuse for this newly produced bundle
- media after conversion

The shared navigation/cancel/error lifecycle is now closed separately in
`../20260721_developer_conversion_lifecycle_current/`. That gate does not
promote other quant profiles or generic JANGTQ/MXTQ conversion.

The source and output contain Qwen vision metadata, but current model
detection deliberately set `forceTextOnly=true` because this freshly affine-
converted hybrid Qwen artifact lacks the independent runtime-verified vision
stamp. The text/agent gate must not be misreported as converted-model VL proof.
That newly observed conversion-to-media boundary remains `PARTIAL` and should
be investigated separately without weakening the safety gate or assuming that
copied vision files equal a working live media route.

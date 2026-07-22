# Qwen3.6 JANGTQ 1.6.16 release checkpoint

Status: `VERIFIED-LIVE_SCOPED / AUTO-REASONING REQUIRED-TOOL PARTIAL`.

Source commits:

- `666c02094e72ae90cf4b8094512048a6b10af69c`: setup and session launch share
  one development source-venv resolver.
- `9cb962acdd364ed98ee7dba7a08d05ec733f234d`: recover a closed Qwen tool block
  whose function opener is missing but whose advertised name and parameters
  are schema-valid.
- `74dadd30c4308fff39d3df4ad0faa529f5e6c9dd`: recover the related live form
  where an empty `<parameter=file_info>` carries the advertised function name.

The parser repairs do not invent missing arguments or accept an unadvertised
tool. Local and remote parser/fallback suites passed `140/140`. Engine
discovery tests passed `11/11` plus typecheck. The remote live files matched
current-source hashes:

```text
fd4e8533169d2b31b626f69fc736c579d283bb1c6546325e6dc028ed8f723e90  vmlx_engine/tool_parsers/qwen_tool_parser.py
54698f7da9e7517bb8e15f9e3e7b81d4a644103eb3ba9949aa3d13a2c7890acb  panel/src/main/engine-manager.ts
fe1fdf2266be78b723de8a351365165ab71fb625a65f390f3eae7046fef83dbb  panel/src/main/sessions.ts
```

## Current live protocol proof

Electron Sessions Start loaded
`dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK` as PID 41929 on port 8007. Startup
reported Qwen tool/reasoning parsers, 10 attention TurboQuant cache layers, 30
native SSM companion layers, q4 native storage, Paged RAM, and Block Disk L2.
This artifact is MXTQ/JANGTQ Hadamard-codebook quantization, not affine JANG or
base MLX MXFP.

With `enable_thinking=false`, `all-protocols-thinking-off.json` passed all 16
requested flows: direct/gateway, Chat/Responses/Anthropic/Ollama, and
stream/non-stream. Every flow executed exactly one `file_info`, exactly one
`run_command`, preserved real results, exact-finaled, used truthful terminals,
and leaked no native control markup.

The fresh Electron full-catalog turn after the parser repair executed both
real tools in order. Its SQLite row retained 74 output tokens, 5,069 prompt
tokens, 3,904 `paged+ssm+disk+tq-native` cached tokens, 0.65 s TTFT, 97.2 tok/s,
two real results, and no warnings. The model copied `/Users/eric/...` as
`/Users//...`; the screenshot preserves that default-sampling synthesis miss
instead of calling the strict-output row a pass.

## Honest remaining boundary

The same explicit Anthropic required-tool request was A/B tested against the
same artifact and template. Thinking On produced reasoning but no schema-valid
tool; Thinking Off emitted the exact `file_info` call. The matrix therefore
proves the explicit direct/tool rail, not universal interleaved
reasoning-plus-tool reliability. No hidden thinking-off retry or fabricated
argument was added. See `anthropic-thinking-on-failure.json`.

Artifacts:

- `all-protocols-thinking-off.json`
- `anthropic-thinking-on-failure.json`
- `electron-two-tool-default-sampling.png`

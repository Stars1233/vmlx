# Bonsai 1-bit partial-prefix cache and Responses continuation

Date: 2026-07-19

Current source: `359ce6b2b64795a3a79bd32d5877935637efb57d`

Verdict: scoped `PASS-LIVE` for the rows named below. The overall model,
cross-parser, signed-app, and release matrices remain `PARTIAL`.

## Artifact truth

The exercised artifact is:

`/Volumes/EricsLLMDrive/jangq-ai/Bonsai-27b-1bit-JANG`

This bundle is **not** JANGTQ/MXTQ. Its committed bundle metadata says:

- `config.json`: `model_type=qwen3_5`, architecture
  `Qwen3_5ForConditionalGeneration`, nested
  `text_config.model_type=qwen3_5_text`, 64 layers, 262144 context, and a
  vision configuration.
- The layer graph repeats three `linear_attention` layers followed by one
  `full_attention` layer: 48 companion-state lanes and 16 ordinary attention
  KV lanes.
- `jang_config.json`: `format=jang`, `profile=JANG_AFFINE_1BIT`,
  `method=jang-affine-discrete`, `actual_bits=1.1128`, storage bits 1, and
  bit widths `[1,4]`.
- `generation_config.json`: sampling defaults temperature 1.0, top-p 0.95,
  and top-k 20.

Do not reuse this result as evidence for JANGTQ/MXTQ Hadamard/codebook or
base MLX MXFP loading. Those are separate runtime formats.

## Real Electron load and settings

The real Session Settings drawer, before launch, showed Auto parser `qwen`,
reasoning `qwen3`, family `qwen3.5`, VLM On, Prefix On, Paged On with block
size 64 and 1000 blocks, Block Disk L2 On, and the live codec label
`TQ8 attention KV + native hybrid state`. Stored quantization remained Auto.

The preview and the real PID argv agreed on:

```text
vmlx-engine serve .../Bonsai-27b-1bit-JANG --host 127.0.0.1 --port 8030
  --is-mllm --continuous-batching --cache-memory-percent 0.15
  --use-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000
  --enable-block-disk-cache
  --block-disk-cache-dir .../live-proof-bonsai-1bit-auto-tq-native-20260716
  --block-disk-cache-max-gb 10 --stream-interval 1
  --tool-call-parser qwen --enable-auto-tool-choice
  --reasoning-parser qwen3 --log-level DEBUG
```

The model was stopped and started with the visible Electron `Stop` and
`Start` buttons after the source repair. The new process was PID 1054. The
visible UI showed `Stop`; `/health` reported healthy, model loaded,
`jangq-ai/Bonsai-27b-1bit-JANG`, model type `mllm`, engine `batched`; and the
live process argv matched the preview. The older MiniMax process had already
exited, so this transition also retained the single-active-model invariant.

The historical hook literal `[Engine Manager] Found in PATH: .../.venv/bin/vmlx-engine`
was not recovered from a persistent Electron-main log file in this run. The
visible Start/PID/Stop state, health, and process argv are current live proof;
the literal log-line sub-row remains `PARTIAL` rather than being invented.

## Partial prefix, resident cache, and process-restart L2

`bonsai-partial-abc.json` used one unique 420-segment shared system prefix and
three distinct sibling user suffixes:

| Turn | Result | Cached tokens/detail | First content |
|---|---|---:|---:|
| A cold | exact `B1P-A-DONE` | 0 | 6.941878s |
| B sibling | exact `B1P-B-DONE` | 6336 `paged+ssm` | 0.894057s |
| C sibling | exact `B1P-C-DONE` | 6336 `paged+ssm` | 0.879039s |

After a visible Electron process replacement, `bonsai-partial-d-restart.json`
returned exact `B1P-D-DONE`, restored 6336 tokens as `paged+ssm+disk`, emitted
six content deltas, and began visible content at 0.695333s.

`bonsai-health-after-d.json` identifies the actual cache implementation:

- native family `qwen3_5`, schema `hybrid_ssm_v1`, cache type
  `hybrid_ssm_typed`;
- attention KV at layer indices 3,7,...,63 plus 48 native companion lanes;
- `attention_kv + ssm_companion_state + async_rederive`;
- block disk 99 hits and 99 `tq_native_hits`, with the restored request marked
  disk-hit, reconstructed, and dequantized;
- q8 TurboQuant storage for attention KV only; live resident encode disabled;
  companion state remains native and restores/rederives at a clean boundary;
- one SSM disk hit, zero SSM misses.

This is direct partial-prefix block reuse from resident memory and from disk
after process restart. It does not replace the separately retained
paged-Off/prompt-disk and forced-small-capacity eviction rows.

## Shared Responses failure and source repair

The pre-fix raw Responses tool loop had a valid first `file_info` call. On the
tool-result continuation the model exhausted its reasoning partition in an
incomplete native `<tool_call>` suffix. The finalizer stripped the suffix but
then chose `full_text` as a content fallback, promoting private reasoning to
`output_text`, suppressing the bounded direct answer pass, and returning
`response.incomplete` with zero progressive content deltas.

This was a shared Responses finalization defect, not a Bonsai quant, cache,
tool schema, or Electron renderer defect.

Commit `359ce6b2b` makes the finalizer retain a rejected reasoning-channel
candidate on the reasoning rail, keeps `accumulated_content` empty, prevents
the later `full_text` reasoning-to-content fallbacks, and re-arms the existing
bounded tools-free answer pass when no valid call and no explicit current-turn
tool intent remain. Source trace: `vmlx_engine/server.py` around lines
20001-20049, 20059-20130, and 20273-20324 at this commit.

The regression
`test_qwen35_responses_incomplete_reasoning_tool_suffix_stays_private` proves
two engine passes, a tools-free direct second pass, progressive output deltas,
no leaked private text, one completed terminal, and no incomplete terminal.

Validation command/result:

```text
pytest tests/test_qwen3_answer_pass_policy.py tests/test_answer_pass_streaming.py tests/test_server.py
147 passed, 3 deselected
```

## Current raw Responses live proof

`bonsai-responses-tool-postfix-events.json` is the complete timed SSE capture
from the Electron-started PID 1054.

- Round 1: first reasoning at 0.652965s, 454 separate reasoning deltas, exactly
  one completed `file_info({"path":"panel/package.json"})`, no visible answer,
  and one `response.completed` at 10.189375s.
- Round 2, using `previous_response_id` and the real function result: first
  reasoning at 0.178861s, 185 reasoning deltas, then 18 content deltas from
  `B` at 3.804007s through ` KB` at 4.126561s. It exact-finaled
  `B1-RAW-RESP-TOOL1-DONE SIZE=5.2 KB` and emitted one
  `response.completed` at 4.154867s.
- No repeated function call, reasoning leak, incomplete terminal, or batch-only
  final content occurred on round 2.

## Current Electron agent loop and visual streaming

Fresh chat row 385 ran with Chat Settings Auto reasoning, Responses wire,
built-in tools On, and blank model-default Max Tokens. It persisted:

- exact content `B1-PATCH-UI-TOOL2-DONE SIZE=5.2 KB`;
- 479 characters of separate reasoning;
- exactly one `file_info({"path":"panel/package.json"})` call and its real
  `Size: 5.2 KB` result;
- no warning;
- 64 cached tokens with `paged+ssm+disk`.

The CDP MutationObserver captured 505 DOM mutations. During final synthesis,
the visible assistant content changed at 5540.9ms `SIZE=`, 5548.4ms `SIZE=5`,
5557.1ms `SIZE=5.`, 5574.3ms `SIZE=5.2`, 5606.0ms `SIZE=5.2 K`, and 5614.5ms
`SIZE=5.2 KB`. That is progressive renderer paint, not a terminal batch.
`bonsai-patched-ui-tool2.png` is the final visible Electron state.

## Remaining boundaries

- Repeat this shared rejected-suffix repair on representative non-Qwen parser
  families; the source is shared, but Bonsai alone is not a global live matrix.
- Retain Bonsai stochastic long-reasoning, long pre-tool, long-context,
  media/video, forced-capacity eviction, and signed-packaged-app rows as
  separate gates. Earlier evidence is not silently promoted by this run.
- Preserve explicit cache/TQ Off tests and the separate paged-Off SSD-only
  partial-prefix campaign; this Auto/q8 row does not substitute for them.
- Overall release status remains `PARTIAL`; no release claim follows from this
  scoped closure.

## Preserved artifacts

- `bonsai-partial-abc.json`
- `bonsai-partial-d-restart.json`
- `bonsai-health-after-d.json`
- `bonsai-responses-tool-postfix-events.json`
- `bonsai-settings-prestart-current.png`
- `bonsai-chat-settings-auto-current.png`
- `bonsai-partial-restart-server.png`
- `bonsai-patched-server-pid1054.png`
- `bonsai-patched-ui-tool2.png`

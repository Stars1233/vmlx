# Laguna-M.1 current-source stream, cache, and settings gate

Date: 2026-07-19

Source under test: `6f7b29bc3` on `reconcile/1.5.68`, pushed to
`origin/codex/live-electron-gates-20260715`.

Model bundle:
`/Volumes/EricsLLMDrive/jangq-ai/Laguna-M.1-JANG_2L`

This is an affine JANG `JANG_2L` Laguna artifact. It is not JANGTQ/MXTQ,
base MLX MXFP, or an MTP model. The copied bundle files in this directory are
the authority for model family, prompt template, quant profile, and generation
defaults.

## Scoped verdict

| Row | Verdict | Current evidence |
|---|---|---|
| Real Electron Sessions Start/load | PASS-LIVE | `laguna-current-start-clicked.png`, `laguna-current-session-db.json`, and `laguna-current-restored-auto-argv.txt` show the real session, PID 13706, port 8015, and `.venv/bin/python3 -m vmlx_engine.cli serve` command before/after the turns. The preceding current-main-process gate records the required `.venv/bin/vmlx-engine` PATH discovery. |
| Bundle-grounded settings | PASS-LIVE | Bundle snapshots plus `laguna-current-chat-settings.png`, session DB, argv, health, and cache stats agree on Laguna/qwen3 reasoning, glm47 tool parser, Auto q4 stored-prefix TQ, paged cache, 64-token blocks, 1000 blocks, and block-disk L2. |
| Electron reasoning/content paint | PASS-LIVE | Natural row 409 has non-empty separated reasoning and content. The three screenshots and row JSON show reasoning growth and final content. Tool DOM trace records progressive final-text lengths rather than one terminal batch. |
| Electron tool loop/history | PASS-LIVE | Row 412 made exactly one real `file_info(panel/package.json)` call, received 5.2 KB, then emitted a visible final sentence. After two Save & Restart cycles, row 415 recalled 5.2 KB without a new tool call and restored 4,980 `paged+disk+tq-native` tokens. |
| Raw Responses stream/non-stream | PASS-LIVE | Separate reasoning deltas, progressive output deltas, one exact tool call, tool-result continuation, and one completed terminal are preserved in `laguna-current-protocol.json`. |
| Raw Chat Completions stream/non-stream | PASS-LIVE | Separate reasoning/content, one exact tool call, continuation, stop finish reasons, and one `[DONE]` per stream are preserved in the same protocol artifact. |
| Explicit cache setting Off | PASS-LIVE | Real UI Save & Restart emitted `--kv-cache-quantization none`; max blocks 4 was honored in argv/health, UI reported TQ OFF and 256-token capacity, and no TQ objects/writes/hits were active. |
| Bounded eviction and L2 partial refault | PASS-LIVE with TQ Off | Max four blocks forced ten L1 evictions and ten disk writes. The oldest 4,538-token prompt refaulted three disk blocks / 192 cached tokens as `paged+disk` and reproduced exact answer `166`. |
| Auto q4 warm stability | PASS-LIVE | Four repeated `paged+disk+tq-native` greedy restores were byte-identical to one another. |
| Auto q4 cold equivalence | PARTIAL / observed mismatch | The first full-precision cold greedy output differs from the stable q4-restored output. Three cache-bypass cold runs and five explicit-None cold/raw-cache runs are byte-identical to the cold baseline. This isolates the difference to lossy q4 stored-KV restoration; it is not hidden with a sampler, prompt, or output rewrite. |
| Laguna latency/long soak | PARTIAL | The natural Electron row decoded at about 23.8 tok/s and the disk-restart recall TTFT was 5.10 s. This gate proves correctness and progressive emission, not the outstanding Laguna performance target or long-agent reliability. |

Overall Laguna verdict: current controlled output/protocol/settings/cache paths
are PASS-LIVE, while q4 cold-byte-equivalence, long-agent reliability, and the
Laguna latency target remain PARTIAL. This is not a release-ready claim.

## Source trace

- `vmlx_engine/server.py:3398-3412` derives tool availability from the
  effective generation tool set and makes `tool_choice="none"` authoritative.
  Both streaming endpoints use it at `server.py:17460-17467` and
  `server.py:19075-19081`.
- `vmlx_engine/utils/turboquant_config.py:143-175` selects q4 for compatible
  uncalibrated full-KV families while retaining Bonsai-specific q8 policy.
- `panel/src/main/sessions.ts:3476-3521` builds paged/max-block arguments,
  preserves explicit `--no-paged-cache`, excludes typed native cache families,
  and forwards explicit q4/q8/none for compatible families.
- `panel/src/renderer/src/components/sessions/SessionSettings.tsx:511-534`
  mirrors the same settings into command preview.
- Test commit `6f7b29bc3` updated the stale streaming-tool source contract to
  assert both endpoints use the shared effective-tool helper. It did not change
  server behavior.

## Electron proof

### Natural reasoning then progressive content

Prompt: explain in exactly two concise sentences why cached RAM prefixes reduce
TTFT while disk restore remains slower.

- Reasoning visibly grew from 1,105 to 2,651 to 3,244 characters.
- Final content was non-empty and coherent.
- Persisted metrics: 760 output tokens, 47 prompt tokens, 23.8 tok/s,
  1.06 s TTFT, 33.1 s total.
- Files: `laguna-current-ui-natural-3s.png`,
  `laguna-current-ui-natural-11s.png`,
  `laguna-current-ui-natural-later.png`, and
  `laguna-current-ui-natural-row.json`.

### Tool turn and post-restart history

- Row 412: two separate reasoning segments, exactly one `file_info`, real
  `panel/package.json` result 5.2 KB, one visible final sentence, no warning,
  9,076 `paged+disk+tq-native` cached tokens, 4.01 s TTFT.
- `laguna-current-ui-tool-dom.json` records visible final-answer lengths
  `30, 33, 37, 45, 56, ... 208` during the paint interval, falsifying the
  previously reported "reasoning freezes, then answer appears all at once"
  behavior for this current Laguna turn.
- Row 415 after the Off/max-four and restored-Auto/1000 restarts: visible
  `The file panel/package.json is 5.2 KB in size.`, no tool call, no warning,
  4,980 `paged+disk+tq-native` cached tokens, 5.10 s TTFT.

Files: `laguna-current-ui-rows.json`,
`laguna-current-ui-tool-final.png`,
`laguna-current-restored-auto-ui-recall.json`, and
`laguna-current-restored-auto-ui-recall.png`.

## Raw API proof

`laguna-current-protocol.json` contains raw timed events and non-stream bodies.
The driver is preserved as `laguna_protocol_probe.py`.

Responses:

- no-tool stream: 234 reasoning deltas, 43 output-text deltas, one completed;
- tool stream: 57 reasoning deltas, one exact `file_info` call, two argument
  deltas, one completed;
- follow-up: 130 reasoning deltas, 10 output-text deltas, tool-derived 5.2 KB,
  one completed;
- no-tool and follow-up non-stream responses both returned HTTP 200 and
  `status=completed` with separated reasoning/message items.

Chat Completions:

- no-tool stream: separated reasoning/content, `finish_reason=stop`, one DONE;
- tool stream: one exact call, `finish_reason=tool_calls`, one DONE;
- follow-up: separated reasoning/content, `finish_reason=stop`, one DONE;
- both non-stream rows returned HTTP 200 with separate `reasoning_content` and
  visible `content`.

## Cache policy and determinism A/B

Auto after the final real UI restart:

- argv: paged cache, block size 64, max blocks 1000, block-disk L2; no explicit
  q-mode, so Auto remains authoritative;
- health/cache: `plain_kv_v1`, `paged_kv`, prefix+paged+block-L2 true,
  `turboquant-q4` stored-prefix K/V, live encode false,
  `uncalibrated_full_kv_storage_tq4`;
- restored namespace: 310 disk blocks and 18,932 L2 tokens before the recall.

Explicit Off/max-four after real UI Save & Restart:

- argv included `--kv-cache-quantization none --max-cache-blocks 4`;
- health reported TQ disabled and raw `plain_kv_v1` storage;
- long-prompt answers were `166`, `181`, `199`; oldest refault returned exact
  `166` with 192 cached tokens from three disk blocks;
- final counters: ten L1 evictions, ten disk writes, 601 L2 tokens, zero TQ.

Greedy determinism:

- cold/bypass baseline repeated byte-identically three times;
- explicit None cold plus four raw `paged+disk` restores repeated
  byte-identically five times and matched the cold baseline;
- Auto q4 first cold answer differed from the subsequent q4 restored answer;
- all four q4 restored answers were byte-identical to each other.

Artifacts: `laguna-current-temp0-repeat.json`,
`laguna-current-temp0-bypass-repeat.json`,
`laguna-current-none-temp0-repeat.json`, and
`laguna-current-none-four-eviction.json`.

## Validation

- Dynamic helper check: 4 passed.
- Focused Python gate: 411 passed, 1 skipped across glm47 parser, streaming
  reasoning, TQ cache/clone, bypass, family/format, tool format, answer-pass
  streaming, and hybrid live TQ.
- Focused panel gate: 21 files / 771 tests passed across settings, cache,
  sessions, migrations, reasoning rendering, stream recovery, and tool history.
- Panel TypeScript typecheck passed.

Logs are preserved in:

- `laguna-current-python-focused-tests-after-contract.txt`
- `laguna-current-panel-focused-tests.txt`
- `laguna-current-panel-typecheck.txt`

## Still required

- Do not close Laguna performance until a measured source-vs-reference runtime
  comparison explains or accepts the approximately 24 tok/s decode rate.
- Do not claim q4 is byte-equivalent to cold full precision. Decide accuracy
  policy explicitly if byte identity is a requirement; explicit user Off works.
- Run longer agent/tool soak and larger partial-prefix eviction coverage before
  closing the family-wide reliability row.
- Bundled Python must be rebuilt before any packaging because current source is
  newer than the packaged runtime.

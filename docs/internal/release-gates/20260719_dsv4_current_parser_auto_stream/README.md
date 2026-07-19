# DSV4 current-source Auto reasoning, parser, streaming, and restart/L2 gate

Date: 2026-07-19 PT

Source head used for the final live pass: `4e723f311`

Bundle: `/Volumes/EricsLLMDrive/dealignai/DeepSeek-V4-Flash-JANG-CRACK`

## Verdict

- Electron Start-button load and main-process engine discovery: **PASS-LIVE**.
- DSV4 Auto reasoning state in the fully relaunched settings drawer: **PASS-LIVE**.
- Electron one-tool loop with separated reasoning, real tool result, non-empty final,
  and no warning: **PASS-LIVE**.
- Raw Responses/Chat reasoning/content/tool/terminal transport: **PASS-LIVE** for
  the current controlled prompts.
- Native DSV4 paged/L2 restart restoration: **PASS-LIVE**.
- Strict synthetic marker fidelity and broad stochastic post-tool factual quality:
  **PARTIAL**. Marker mutation and one weak-prompt hallucination are retained; no
  parser/endpoint/artifact cause is claimed without the pending same-artifact
  reference-runtime A/B.
- Overall release: **BLOCKED** by the previously recorded stale bundled-Python copy
  and other open matrix rows.

## Source trace

Commit `4e723f311` makes the panel declare the DSV4 family capabilities used by the
runtime (`dsml`, `deepseek_r1`, thinking/instruct support, `high`/`max`, model-default
thinking) and renders an explicit `Auto` button whenever the chat override is absent.
The exact patch is preserved in `source-trace.txt`.

Relevant source locations at this head:

- `panel/src/main/model-config-registry.ts:157-168`
- `panel/src/renderer/src/components/chat/ChatSettings.tsx:438-480`
- `panel/tests/chat-settings-compatibility.test.ts`
- `panel/tests/model-config-registry.test.ts`

The bundle summary proves this tested artifact is affine JANG, not JANGTQ/MXTQ or
base MLX MXFP: `model_type=deepseek_v4`, `weight_format=affine`, 43 layers, global
4-bit affine plus 2-bit routed experts. No MTP capability is inferred.

## Real Electron proof

The first attempted relaunch omitted the project venv and logged
`ModuleNotFoundError: vmlx_engine` / `Engine Manager Not installed`; it is retained as
`electron-wrong-path.log` and is not counted.

The corrected current-source launch used the persistent dev profile and CDP 9335.
`electron-current.log` contains:

```
[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine
```

The real Sessions-card Start action launched PID 8882 on port 8012 with `dsml`,
`deepseek_r1`, DSV4 native composite prefix cache, paged cache, and block disk cache.
Before any chat turn, `/health` reported `model_loaded=true`, about 99.7 GB active
memory, and 2,818 pre-existing L2 block tokens. This is the eager-load proof for this
row, not a curl-only launch.

After a full Electron main-process restart, the settings screenshot and DOM query
show `Auto` with selected `bg-primary` styling and Instruct/Reasoning/Max unselected.
Built-in tools and File/Search categories were checked.

The fresh Electron prompt requested one `file_info(panel/package.json)`. SQLite row
406 records:

- two separate reasoning rails totaling 121 characters;
- exactly one OAI function call with exact path `panel/package.json`;
- the real result `Size: 5.2 KB`;
- visible final `The file size is 5.2 KB.`;
- `warnings_json=null`;
- 1,125 prompt tokens, 94 output tokens, 1.41 s TTFT.

The screenshot `electron-restart-tool-final.png` visibly shows both Reasoning rails,
the Info tool card, and the non-empty answer.

## Raw API proof and retained negatives

`endpoint-identical-prompt-ab.json` compares identical prompts at temperature 0:
Responses and Chat produce byte-identical outputs with reasoning/content on separate
delta rails, both with and without tool schemas. This excludes an endpoint-specific
duplicate-content re-emission for the controlled row.

`protocol-events-tool2.json` shows both APIs emit one schema-valid tool call, consume
the real result, stream post-tool reasoning and 17 content deltas, and emit completed
terminals. Both mutate the requested synthetic marker in the same way, placing the
mutation before endpoint emission but not yet assigning it to artifact or integration.

`required-tool-ab.json` retains the prompt/sampling sensitivity: three greedy runs
used the correct path; three bundle-default stochastic runs duplicated `panel/` for
one prompt wording. The Electron prompt at the same bundle-default sampling used the
correct path. `protocol-events.json` retains the weak post-tool prompt that caused a
hallucinated package manifest. These are quality failures, not hidden.

## Cache truth after process restart

`health-after-restart-tool.json` reports:

- `engine_path=dsv4`, `DSV4BatchGenerator`;
- 2 scheduler/block-disk hits;
- 3,173 L2 block tokens on disk and 867 RAM/L1 indexed tokens;
- 3 new disk writes;
- `tq_native_enabled=false`, zero generic TQ writes/hits.

This proves the native DSV4 composite cache restored from L2 across the visible
process replacement. It does not claim generic TurboQuant for DSV4; the architecture
owns SWA plus CSA/HCA compressed-pool state and must stay on its native codec.

## Tests

- Python focused DSV4/parser/reasoning/terminal/cache selection: 329 passed.
- Panel registry/settings selections: 100 passed.
- Panel TypeScript typecheck: passed.

## Evidence inventory

- `bundle-config-summary.json`
- `source-trace.txt`
- `python-focused-tests.txt`
- `panel-focused-tests.txt`
- `panel-typecheck.txt`
- `electron-wrong-path.log`
- `electron-second-bind-failure.log`
- `electron-current.log`
- `electron-relaunch-home.png`
- `electron-auto-settings.png`
- `electron-restart-tool-final.png`
- `electron-restart-tool-row.json`
- `health-before-restart-turn.json`
- `health-after-restart-tool.json`
- `health-first-load.json`
- `health-first-run-after.json`
- `electron-first-no-tool.png`
- `electron-first-tool.png`
- `electron-first-db-rows.json`
- `electron-first-no-tool-dom.json`
- `electron-first-tool-dom.json`
- `required-tool-ab.json`
- `endpoint-identical-prompt-ab.json`
- `protocol-events.json`
- `protocol-events-tool2.json`
- `electron-engine-relevant.log`

## Required follow-up

1. Run a controlled same-artifact reference-runtime A/B with identical template,
   sampling, prompt, and result history before attributing strict marker mutation or
   long-answer quality to the official bundle.
2. Keep the broader constrained-string/stochastic/long-quality row PARTIAL.
3. Refresh bundled Python with `panel/scripts/bundle-python.sh` at the release cutoff,
   rerun the full bundle verifier, and repeat a signed-app smoke before release.

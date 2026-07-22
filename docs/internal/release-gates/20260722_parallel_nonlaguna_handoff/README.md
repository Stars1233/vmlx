# Parallel non-Laguna handoff for release finalizer

Last updated: 2026-07-21, America/Los_Angeles.

Eric explicitly requested this duplicate/parallel lane stop after writing a
complete handoff. No further implementation, model runs, packaging, signing,
notarization, tagging, or publishing should be attributed to this lane after
this document.

This handoff is for the Python/Electron checkout:

`/Users/eric/mlx/vllm-mlx-release-1.6.13`

Branch:

`codex/postrelease-ui-drawers-20260720`

## Current synchronized source state

- Current checkout, upstream tracking ref, and GitHub branch all matched at
  `2ad94974f0caea8bd29211a46c4eb5a8af762985` when this handoff was written.
- This lane's scoped fix is pushed in ancestor commit
  `ee227bcdccdfdbaf63e3113fb82ae5e2e6a65806`
  (`fix(lfm): bind natural file tool paths`).
- The local second-machine checkout `/Users/eric/mlx/vllm-mlx` fetched the
  branch ref through `ee227bcdc`; its dirty worktree was not modified.
- The active remote checkout was intentionally left with only other-agent or
  local dependency state:
  - modified `.agents/LOG.md`
  - modified `.agents/STATUS.md`
  - untracked `panel/node_modules` symlink
- This lane did not stage, reset, revert, or overwrite those paths.

## What this lane changed

### LFM2 natural file-tool binding

Files:

- `vmlx_engine/api/tool_calling.py`
- `tests/test_tool_prompt_fallback.py`

Root cause: an explicit natural request such as
`Use file_info exactly once to inspect panel/package.json` did not contain the
literal phrase `path panel/package.json`, so the LFM native fallback example
remained `file_info(path='VALUE_HERE')`. The live model copied that server
placeholder and emitted the wrong schema-valid argument.

Fix: bind a candidate only when all of the following are true:

1. the request explicitly names the selected tool;
2. the parameter is path-like (`path`, `file_path`, or `filepath`);
3. the path follows a file-oriented verb (`inspect`, `examine`, `stat`,
   `read`, or `check`);
4. the candidate is actually path-like (contains `/` or `.`, or begins with
   `~`).

The last condition prevents a prompt such as `inspect it` from binding the
pronoun `it` as a file path.

Focused current-source test:

- `tests/test_tool_prompt_fallback.py`: **25 passed**
- `git diff --check`: clean before commit

## Live LFM model/API proof

Model:

`/Volumes/EricsLLMDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK`

Isolated server:

- `127.0.0.1:8051`
- source loaded from the release checkout via `PYTHONPATH`
- `--stream-interval 1`
- tool parser `lfm2`
- reasoning parser `qwen3`

The isolated server was stopped after proof; no listener remained on port
8051. The shared Electron app/Laguna session owned by the other agent was not
touched.

### Runtime architecture observed

- 24 state layers total
- 6 attention layers using `TurboQuantKVCache`
- 18 native SSM/array layers
- stored attention prefix state: native q4 TurboQuant
- SSM companion: native full precision with re-derive
- paged cache and block-disk L2 enabled

Warm runtime logs showed block-aligned paged hits, partial hybrid
reconstruction, q4 attention-state reuse, and synchronous/deferred SSM
companion re-derive. This was not a process-restart disk-only proof, so do not
promote it to that stronger claim.

### Auto reasoning, field omitted

Chat Completions request omitted `enable_thinking` entirely:

- 84 `delta.reasoning_content` chunks from 0.081486s through 0.422410s
- 2 `delta.content` chunks from 0.434668s through 0.438680s
- visible content: `42.`
- terminal `finish_reason=stop` at 0.445300s
- zero raw `<think>`, `</think>`, `[THINK]`, or `[/THINK]` markers

This is current live evidence that LFM Auto resolves to thinking On. Explicit
Off separately returned HTTP 400 with the advertised no-instruct-mode
contract.

### Explicit reasoning and progressive visible output

Chat Completions:

- 73 reasoning deltas
- 6 content deltas
- visible answer `The product is 323.`
- terminal `stop`

Responses:

- 120 reasoning-summary deltas
- 5 output-text deltas
- visible answer `Spring Summer Autumn Winter`
- terminal `response.completed`

### Chat tool loop

- initial turn: 81 reasoning deltas
- exact structured call:
  `file_info({"path":"panel/package.json"})`
- terminal `tool_calls`
- tool-result continuation: 76 reasoning deltas, then 11 content deltas
- exact visible answer: `LFM-TOOL-CONTINUATION-PASS`
- no second tool call; terminal `stop`

### Responses tool loop

- initial turn at `max_output_tokens=2048`: 178 reasoning deltas
- function arguments arrived progressively in two deltas
- exact completed arguments: `{"path":"panel/package.json"}`
- terminal `response.completed`
- usage: 239 input, 416 output, 235 cached input tokens
- cache detail: `paged+ssm+tq-native`
- function-output continuation: 72 reasoning deltas, then 10 output-text
  deltas
- exact visible answer: `LFM-RESPONSES-TOOL-PASS`
- no repeated tool; terminal `response.completed`
- continuation cache: 162 cached of 166 input tokens,
  `paged+ssm+tq-native`

The same initial Responses tool request at an explicit 1,024-token cap emitted
a valid completed tool call but honestly ended `response.incomplete` with
reason `max_output_tokens`. It completed at 2,048. Do not describe the 1,024
row as a completed response or hide the budget boundary.

Full sanitized evidence:

- `docs/internal/release-gates/20260722_lfm_reasoning_tool_stream/README.md`
- `docs/internal/release-gates/20260722_lfm_reasoning_tool_stream/proof-summary.json`

## Required non-Laguna reasoning work still open

The read-only family audit is preserved at:

`docs/internal/release-gates/20260722_global_reasoning_source_audit/README.md`

The relevant source files did not change between the audited `e16a3dac1` and
the current handoff head. Release-blocking rows remain:

1. Ollama streaming must apply MiniMax-M3 `thinking_mode` and Mistral4
   `reasoning_effort` normalization like Chat/Responses/Anthropic.
2. Anthropic streaming must correctly close and reindex a visible text block
   before opening late reasoning; Gemma4 visible -> late thought is the
   required adversarial case.
3. Ollama streaming `think:false` must strip historical private reasoning as
   Chat, Responses, and Anthropic already do.
4. Generic ThinkParser and MiniMax-M3 need partial-marker holdback so split
   `<think>`, `[THINK]`, and `<mm:think>` delimiters cannot leak across token
   boundaries.
5. Reasoning-only tool markup needs Chat and Responses live proof that private
   residue cannot become visible content.

Two isolated helper worktrees were created but interrupted before any edit or
commit because Eric requested this pause:

- `/Users/eric/mlx/vllm-mlx-nonlaguna-reasoning-parity-20260721`
- `/Users/eric/mlx/vllm-mlx-parser-holdback-ee227`

Both remained clean at `ee227bcdc`. They may be safely reused or removed by
the finalizer after checking them again.

Minimum remaining live matrix:

1. MiniMax-M3 Ollama stream/non-stream Auto/On/Off plus required tool.
2. Mistral4 Ollama stream/non-stream On/Off and effort high/none.
3. Gemma4 Anthropic visible -> late thought -> final/tool event balance.
4. Qwen3/Bonsai two-turn Ollama Off history stripping plus required tool.
5. DSV4 Responses Auto/On/Off plus required DSML tool.
6. MiniMax-M2 streaming On/Off plus tool.

LFM Chat/Responses are live-proven in this handoff, but LFM
Anthropic/Ollama/Electron rows remain unrun.

## Release packaging blockers

The source-only packaging audit is preserved at:

`docs/internal/release-gates/20260722_release_guard_source_audit/README.md`

Do not package from the current ad-hoc bundled Python or dependency tree.
Release blockers include:

1. bundling does not reject untracked JANG/vMLX package files or pin exact
   expected source SHAs;
2. Sequoia and Tahoe are not tied to one immutable provenance manifest;
3. packaged import gates do not require interpreter/package paths to remain
   inside the app;
4. raw extraResource `vmlx_engine/**` parity is not fully checked;
5. missing JANG source can be SKIP instead of FAIL;
6. `codex_ui_only` can bypass the offline manifest;
7. remote noninteractive PATH lacks bare `node`;
8. current `panel/node_modules` is a symlink into a sibling checkout.

## JANG release source

- GitHub `origin/main`: `801209c13c189ebb8fb4d1596748a336f568da38`
- Clean release-prep worktree:
  `/Users/eric/jang-release-prep-20260721`
- That worktree was clean at exact `801209c` when checked.
- Do **not** bundle from `/Users/eric/jang`: it remained at dirty
  `ca75f0cb` with tracked and extensive untracked changes.

Earlier independent integration proof for JANG `801209c`:

- 573 passed / 37 skipped across the selected suite
- package build succeeded
- prior focused quant/runtime review: 66 passed
- semantic quantization remained `{2,3,4,5,6,8}`; 1-bit remained
  storage-only

No large-model JANG run was performed in that source-only review.

## Exact release-finalizer order

1. Re-read, in order:
   - `docs/internal/release-gates/20260722_active_worklist/README.md`
   - this handoff
   - `../20260722_lfm_reasoning_tool_stream/README.md`
   - `../20260722_global_reasoning_source_audit/README.md`
   - `../20260722_release_guard_source_audit/README.md`
   - `docs/internal/ISSUE-LEDGER.md`
   - `docs/internal/PYTHON_ENGINE_MODEL_GATE_MATRIX.md`
   - `docs/internal/CACHE-DEFAULTS-UI-WIRING-MATRIX.md`
2. Recheck branch HEAD/upstream/GitHub and every dirty path before editing.
3. Close the remaining global reasoning source blockers and focused tests.
4. Run the minimum raw-wire model matrix above on current source.
5. Run real Electron Start-button/multi-turn/tool/streaming proof on the final
   source head, including visual reasoning/content separation and complete
   terminal UI state.
6. Run complete Python and panel suites, panel typecheck, production build,
   and diff hygiene. Focused suites do not close this row.
7. Add and execute the packaging source-preflight/provenance guards.
8. Bump all version surfaces consistently to the selected next checkpoint
   (expected next patch from 1.6.14 is 1.6.15 unless Eric directs otherwise).
9. Bundle from the clean JANG `801209c` worktree and a clean, pushed vMLX
   release head. Use an owned dependency tree, not the sibling symlink.
10. Build both Sequoia and Tahoe from the same immutable source manifest;
    sign, notarize, staple, verify wheel tags, and install-smoke each DMG.
11. Verify bundled interpreter and all engine/JANG imports resolve from inside
    each installed app under poisoned/cleared Python environment variables.
12. Only then tag and publish GitHub/PyPI/update feeds. Do not use
    `codex_ui_only` for the public checkpoint.

## Explicit proof-gate status for this lane

- Electron Start-button load: **N/A** — intentionally not run because another
  agent owned the shared Laguna UI session.
- Bundle-grounded autodetection: **PASS for isolated LFM only** — live startup
  and capabilities/log configuration named above.
- Non-empty visible output: **PASS for every LFM generation retained as a
  passing row**; the 1,024-token incomplete row is explicitly classified.
- Verbatim cross-turn reasoning comparison: **PARTIAL** — prompts produced
  different reasoning; no byte-identical stale replay was observed, but this
  lane did not retain a formal full-text cross-turn diff artifact.
- Separate reasoning/content deltas: **PASS for LFM Chat and Responses raw
  API**; Electron/Anthropic/Ollama are unverified here.
- Required tool execution/continuation: **PASS for LFM Chat and Responses**.
- Raw marker/leak scan: **PASS for the six retained LFM captures**; zero
  `<think>`, `</think>`, `[THINK]`, `[/THINK]`, or `<mm:think>` output markers.
- Three-turn recall/history: **N/A** — only the two-stage tool loops were run.
- Cold/warm/paged/SSM/TQ cache: **PARTIAL** — warm paged/hybrid/q4/SSM reuse
  observed; restart-from-disk and paged-Off disk-only rows were not run.
- Documentation: **PASS** — three focused gate documents plus this handoff.
- Commit/push synchronization: **PASS for `ee227bcdc`**; current branch and
  GitHub were later synchronized at `2ad94974f` by the other agent.
- Second computer-use verifier: **N/A** — no Electron work was performed in
  this lane.

Overall status: `PARTIAL / PAUSED / NOT RELEASE-READY`. The LFM
Chat/Responses defect is fixed and live-proven, but the named global reasoning,
Electron, full-suite, provenance, packaging, signing, notarization, and publish
gates remain for the release finalizer.

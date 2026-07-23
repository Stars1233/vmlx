# v1.6.17 release-candidate smoke

Date: 2026-07-23 (America/Los_Angeles)

Status: `DEV UI+API+PANEL PASS / BUNDLE+SIGNED APP OPEN`.

## Source boundary

- Repository:
  `/Users/eric/mlx/vllm-mlx-r17-consolidation`.
- Branch:
  `codex/v1.6.17-consolidation-20260723`.
- Raw direct/gateway API and agentic protocol evidence was captured at
  `aa49b699d`.
- The only change from `aa49b699d` to the current smoke head `8eb8468bd` is
  the panel rejected-tool-control-markup correction and its tests:
  `panel/src/main/ipc/chat.ts`,
  `panel/src/shared/responsesStreamRecovery.ts`, and
  `panel/tests/responses-stream-recovery.test.ts`. No engine/API source changed
  between those heads.
- The post-fix Electron evidence was captured from the exact main bundle built
  at `8eb8468bd`.
- This is development-app evidence. It does not replace bundled-Python,
  signed-DMG, installed-app, Gatekeeper, or notarization proof.

## Live Electron proof

The isolated development app used
`/Users/eric/.vmlx-r17-consolidation-dev` and CDP port `9335`. The real
Electron Start button loaded the exact bundle
`JANGQ-AI/Hy3-JANG_2K-MTP` as PID `47532` on port `8010`.

The development log resolved the project engine:

```text
[Engine Manager] Found development project venv:
/Users/eric/mlx/vllm-mlx-r17-consolidation/.venv/bin/python3
(vmlx_engine 1.6.16)
```

Live health before the turns showed:

- model loaded and healthy;
- `hunyuan` tool parser and `qwen3` reasoning parser in argv;
- bundle-derived sampling `temperature=.9`, `top_p=.9`, `top_k=-1`
  (effective `0`), and `min_p=.05`;
- q4 TurboQuant stored-prefix KV;
- Paged RAM and Block Disk L2 both enabled;
- native HY3 MTP tensors detected but acceleration validation-blocked, with
  safe autoregressive generation active.

A fresh chat then completed three UI turns:

1. no-tool reasoning plus visible currency and TeX;
2. exactly one real `file_info(panel/package.json)` execution and real
   `5.2 KB` continuation;
3. no-tool history recall of the exact real path/size.

Observed UI results:

- every turn showed a separate Reasoning rail;
- the tool turn showed two distinct reasoning phases, one successful Info card,
  and the exact visible continuation;
- the first answer contained one real `.katex` node and visually rendered
  `9 × 6 = 54`;
- literal currency `$43` remained currency;
- no `<think>`, tool XML, argument markers, parser residue, or replacement
  characters were visible;
- generation controls appeared during each live turn and returned only after
  stream completion;
- visible answers were non-empty and exact;
- cache details were visible in the UI:
  `82 paged+tq-native`, `274 paged+disk+tq-native`, and
  `192 paged+tq-native` cached prompt tokens.

The model was stopped with the real Electron Stop button after evidence
capture. Gateway health then reported all saved backends stopped.

Evidence:

- `r17-electron-postfix-8eb8468bd.json`
- `r17-electron-postfix-turn-1.png`
- `r17-electron-postfix-turn-2.png`
- `r17-electron-postfix-turn-3.png`

## Raw API streaming proof

While the same Electron-started PID was loaded, direct port `8010` and gateway
port `8091` each exercised streaming:

- OpenAI Chat Completions;
- OpenAI Responses;
- Anthropic Messages;
- Ollama Chat.

All eight streams returned HTTP 200, emitted non-empty private reasoning
separately from progressive visible content, preserved `$43` and `9×6=54`,
contained no visible control markers, and emitted a protocol-native terminal.

Exact wire surfaces included:

- Chat: `delta.reasoning_content` before `delta.content`;
- Responses: `response.reasoning_summary_text.delta` before
  `response.output_text.delta`;
- Anthropic: `thinking_delta` before `text_delta`;
- Ollama: `message.thinking` before `message.content`.

Seven rows matched the requested punctuation exactly. Direct Chat omitted only
the final period while preserving the entire requested sentence. The retained
artifact therefore honestly has `pass:false` under its strict exact-string
aggregate even though every structural/streaming/transport check passed
8/8. This variation is not rewritten or hidden.

Raw stream payloads are not committed because they contain private reasoning.
`r17-raw-wire-aa49b699d.json` retains raw-stream SHA-256 values, reasoning
lengths/hashes, normalized event kinds, content, terminal events, parse
results, and per-row checks.

## Three-round agentic API proof

Direct and gateway Chat/Responses/Anthropic/Ollama each completed one
three-round streaming harness:

1. exact `file_info(panel/package.json)`;
2. exact `run_command(pwd)`;
3. exact final synthesis using the real results.

All eight flows passed:

- exact tool names and schema-valid arguments;
- real tool results;
- no visible prose during the two tool rounds;
- non-stale separate reasoning;
- progressive exact final content;
- truthful protocol-native terminals;
- final `SIZE=5.2 KB`;
- final `PWD=/Users/eric/mlx/vllm-mlx-r17-consolidation`.

`r17-agentic-stream-aa49b699d.json` retains event timing and reasoning hashes,
not private reasoning text.

## Panel and source suites

At `8eb8468bd` on the M5 Max:

- panel: `86` files, `2491 passed`, `3 skipped`;
- TypeScript `tsc --noEmit`: pass;
- production `electron-vite build`: pass;
- production output contains the KaTeX font assets.

Evidence:

- `r17-panel-full-8eb8468bd.log`
- `r17-panel-typecheck-8eb8468bd.log`
- `r17-panel-build-8eb8468bd.log`

The complete Python source suite was already rerun at engine-corrected head
`107be113b`: `6407 passed, 96 skipped, 93 deselected`. The only manually
deselected row is the intentional bundled-Python SHA integrity gate. The
subsequent changes through `8eb8468bd` are documentation and panel-only.

## Remaining release gates

The checkpoint is still not release-ready until all of these complete on the
versioned source head:

1. bump all source/package versions to `1.6.17`;
2. add and pass a 1.6.17 scoped preflight without pretending the retained
   campaign PARTIAL rows are closed;
3. rebuild bundled Python from the frozen source and pass
   `verify-bundled-python.sh`;
4. rerun the complete Python suite including the bundled-Python integrity row;
5. build Developer-ID-signed Sequoia and Tahoe DMGs;
6. submit both to Apple, wait for acceptance, staple, validate, run codesign
   and Gatekeeper checks, and record SHA-256;
7. install-smoke each signed artifact, including real Electron Start/Stop,
   visible chat, and raw API streaming;
8. only then tag, publish the GitHub release/PyPI artifact, and update
   `latest.json` plus public feeds.

Retained broader family/media/cache/stress rows remain follow-up work after
this usable checkpoint and must remain marked `PARTIAL`.

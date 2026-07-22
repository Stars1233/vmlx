# Laguna mixed-SWA L2 tool gibberish gate — 2026-07-22

## Verdict

PARTIAL for release: the specific Laguna warm-cache gibberish regression is fixed and live-proven on the CLI/API path, but Electron visual proof and full checkpoint packaging are still separate gates.

## Failure observed

User-visible Laguna row in `/Users/eric/.vmlx-v1613-responsive-dev/chats.db` produced:

- visible content: `[Generation interrupted]`
- reasoning rail: random token gibberish beginning `abarCrumbSEL HaberfeldPitch...`
- metrics: `cachedTokens=64`, `cacheDetail=paged+tq-native`, `finish=length` class behavior
- tool status: only speculative `phase=generating`; no completed tool call

An older row also showed visible gibberish plus `[full] Negative dimensions not allowed`, so this was treated as a release blocker, not a UI cosmetic issue.

## Root cause classification

Live A/B isolated the failure to stale L2/paged block reuse:

- pre-fix fresh/salted request produced a valid structured `file_info` tool-call delta
- pre-fix first warm request restored a stale 64-token L2 block and produced Laguna reasoning gibberish
- pre-fix second warm request succeeded after the runtime wrote a new 181-token clean prompt-boundary block

That points to old block-disk entries crossing the newer mixed-SWA clean prompt-boundary + generation-prompt side-key contract.

## Source fix

Commit: `1c8ef8ecd fix(cache): invalidate stale mixed-swa l2 blocks`

Files:

- `vmlx_engine/prefix_cache.py`
  - bumped `PAGED_CACHE_SCHEMA_VERSION` to `paged_n1_keys_v9_generation_prompt_sidekey`
  - documents that v8 blocks can poison Laguna/other mixed-SWA tool-selection turns after generation prompt side-key and clean prompt-boundary changes
- `tests/test_generation_prompt_cache_key.py`
  - adds a regression that current schema contains `v9` and `generation_prompt_sidekey`

## Source tests

Remote checkout: `/Users/eric/mlx/vllm-mlx-release-1.6.13`

Command:

```sh
/Users/eric/mlx/vllm-mlx/.venv/bin/python -m pytest tests/test_generation_prompt_cache_key.py tests/test_streaming_reasoning.py -q
```

Result: `141 passed in 1.90s`

## Live proof

Model:

```text
/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L
```

Server route:

```text
127.0.0.1:8008
tool parser: glm47
reasoning parser: deepseek_r1
cache: paged + block disk L2 + mixed-SWA native TQ q4 storage policy
```

Fresh v9 namespace proof:

- log: `/tmp/laguna_cli_repro_v9_20260722.log`
- block store initialized at `/Users/eric/.cache/vmlx-engine/block-cache/76568903fd2e` with `entries=0`
- fresh/salted row: valid `file_info({"path":"panel/package.json"})`
- warm row: valid `file_info({"path":"panel/package.json"})`
- second warm row: valid `file_info({"path":"panel/package.json"})`
- no gibberish flagged

Restart-from-disk proof:

- log: `/tmp/laguna_cli_repro_v9_restart_20260722.log`
- evidence JSON: `laguna_tool_repro_v9_restart_l2.json`
- restarted block store initialized at `/Users/eric/.cache/vmlx-engine/block-cache/76568903fd2e` with `entries=3`
- first unsalted request after restart hit L2:
  - `disk_hits=3`
  - `disk_promotion_hits=3`
  - `tokens_saved=362` by end of run
  - log line: `worker reconstructed 3 block(s) from L2`
- all rows emitted valid `file_info({"path":"panel/package.json"})` tool-call deltas
- no gibberish flagged

Electron UI proof:

- screenshot: `laguna-ui-tool-v9-restart-pass.png`
- persisted DB row: `laguna-ui-tool-v9-restart-row.json`
- health snapshot: `laguna-ui-tool-v9-health.json`
- model loaded by the real Electron UI Start button into running `Stop` state
- user prompt: `[LAG-S21-UI-TOOL-V9-RESTART] ... file_info ... panel/package.json`
- final visible answer: `LAG-S21-UI-TOOL-V9-RESTART-DONE SIZE=5.2 KB`
- tool loop:
  - `phase=calling`, `toolName=file_info`, `detail={"path": "panel/package.json"}`
  - tool result reported `Size: 5.2 KB`
  - `phase=done`
- metrics:
  - `promptTokens=684`
  - `cachedTokens=64`
  - `cacheDetail=paged+disk+tq-native`
  - `ttft=1.20`
  - `totalTime=3.4`
- no warnings and no gibberish in the persisted row

## Separate blocker observed during UI proof

While the UI Laguna session was running on port 8008, a separate LFM2 engine process was also still alive on port 8051. This means the global "one model only" lifecycle/swap gate is not closed by this proof and must remain a release-check item.

## Remaining gates before release

This does not close:

- full API protocol matrix
- global one-model-only lifecycle/swap cleanup
- packaged Sequoia/Tahoe DMG signing/notarization/Gatekeeper install smoke

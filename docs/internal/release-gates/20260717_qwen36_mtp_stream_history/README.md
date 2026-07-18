# Qwen3.6 MTP Responses streaming/history gate - 2026-07-17

Status: **PARTIAL / RELEASE BLOCKED**.

This row covers the actual requested MTP artifact:

`/Volumes/EricsLLMDrive/dealignai/Qwen3.6-27B-MXFP8-CRACK-MTP`

It does not cover Bonsai, DSV4 Flash, Gemma4 rotating SWA, Laguna, Step,
Nemotron, MiniMax, or release packaging.

## Source trace

- `vmlx_engine/server.py`
  - keeps Qwen3.5/3.6 bounded visible-answer retries on the direct rail by
    passing `enable_thinking=false` through both the top-level request kwargs
    and `chat_template_kwargs`.
  - covers both Chat Completions and Responses streaming answer-pass paths.
- `tests/test_qwen3_answer_pass_policy.py`
  - adds Qwen3.5-family regressions for blank/Auto output-budget rows.
  - asserts the retry streams visible content deltas rather than finalizing as
    reasoning-only content.
- `panel/src/shared/toolHistoryReplay.ts`
  - treats a persisted Responses reasoning-only assistant row as an empty
    assistant boundary, not as replayable hidden reasoning text.
  - preserves ordered reasoning for real tool-loop rows.
- `panel/tests/tool-history-replay.test.ts`
  - pins the no-stale-hidden-replay behavior.

## Focused tests

```text
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m pytest tests/test_qwen3_answer_pass_policy.py -q
5 passed

cd /Users/eric/mlx/vllm-mlx/panel
npm test -- --run tests/tool-history-replay.test.ts
4 passed
npm run typecheck
completed
```

## Live API evidence

Evidence file:

- `q36mtp-api-stream-proof.json`

Server argv included:

```text
--tool-call-parser qwen
--enable-auto-tool-choice
--reasoning-parser qwen3
--cache-memory-percent 0.15
--use-paged-cache
--paged-cache-block-size 64
--max-cache-blocks 1000
--enable-block-disk-cache
--block-disk-cache-max-gb 10
--stream-interval 1
--native-mtp-depth 3
--native-mtp-sampling-policy deterministic-defaults
```

Health/source-trace from the live server:

- `mtp.status=native_runtime_active`
- `mtp.family=qwen3_5`
- `mtp.effective_depth=3`
- `mtp.runtime_scope=text+vl`
- `mtp.vl_runtime_available=true`
- JANG loader log: hybrid model with 16 attention layers and 48 SSM layers.
- JANG loader log: TurboQuant storage codec is 4-bit attention KV only, policy
  `qwen_hybrid_attention_kv_storage_tq4`.

API stream result:

- Cold Responses stream:
  - exact visible text: `Q36-MTP-API-STREAM-OK`
  - 73 reasoning deltas
  - 7 separate visible content deltas
  - first content delta at 7512 ms, completed at 8398 ms
- Warm Responses stream:
  - exact visible text: `Q36-MTP-API-STREAM-OK`
  - 78 reasoning deltas
  - 7 separate visible content deltas
  - usage reported `cached_tokens=41`, `cache_detail=paged+ssm`
- Post-proof health:
  - block disk cache `tq_native_enabled=true`
  - `tq_native_writes=1`
  - `tq_native_hits=9`
  - SSM companion L2 enabled separately.

## Red / missing evidence

- The live Electron dev app could not be relaunched from SSH after restarting
  the panel main process. CDP `127.0.0.1:9335` was not reachable after attempts
  via direct `electron-vite`, `launchctl asuser`, and `open`.
- Therefore there is no current live visual Electron proof for the
  `toolHistoryReplay.ts` fix.
- The earlier visible UI rows before this source change remain red evidence:
  repeated blank assistant content with non-empty reasoning, and stale prior
  prompt text replayed into following Qwen3.6 turns.

Current release verdict for this row: **PARTIAL**. Source and API evidence are
present; required live Electron evidence is missing.

## 2026-07-17 21:00+ update — Electron relaunch blocker RESOLVED, visual proof in flight

- The SSH relaunch failure above is fixed. Root causes: (1) prior `pkill
  electron-vite` left the old Electron process holding port 9335 (`bind()
  failed: Address already in use` → CDP served the OLD build); (2) the SSH
  launch env lacked the Python venv on PATH, so the panel logged
  `[Engine Manager] Not installed` (vmlx-serve undetectable → model loads
  broken).
- Working relaunch (proven live 20:53): kill `electron-vite dev`, any
  `remote-debugging-port=9335` process, and `panel/node_modules/electron/dist/
  Electron.app` processes; confirm 9335 free; then launch with
  `PATH=/Users/eric/mlx/vllm-mlx/.venv/bin:$HOME/.local/bin:$HOME/.local/node/bin:$PATH`
  and `VMLINUX_USER_DATA_DIR=/Users/eric/.vmlx-v1611-cachefix-dev`.
- Verified after relaunch: `[STARTUP] Using vMLX userData override:
  /Users/eric/.vmlx-v1611-cachefix-dev`, `[Engine Manager] Found in PATH:
  /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`, CDP 9335 answering,
  Qwen3.6-27B-MXFP8-CRACK-MTP server alive on 8032 (health standby_soft,
  model_loaded=true).
- NEW P0 logged from pre-patch UI evidence (20:35-20:38): ledger row
  `Q36MTP-UI-REASONING-REPLAY` — later NOTOOL turns rendered byte-identical
  replayed reasoning from the first NOTOOL turn with EMPTY visible answers
  while timing stats stayed live. Patched-build retest (5-turn incl. tool turn
  + both NOTOOL prompts, screenshots + verbatim DOM reads to `ui-proof/`) is
  running via Codex gpt-5.6-sol xhigh computer-use over CDP.
- Box codex auth is revoked (`refresh_token_invalidated`) — box-side codex
  runs blocked until `codex login` on the box; current run drives the box UI
  from max2 through an SSH tunnel of 9335.
- Row status remains **PARTIAL** until that visual evidence lands.

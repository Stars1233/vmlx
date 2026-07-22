# Single-model orphan process prune — 2026-07-22

## Verdict

`PASS-LIVE_SCOPED` for the specific one-model lifecycle bug where a live
detected `vmlx-engine` process survived even though its persisted session row was
stopped or stale.

Overall release status remains `PARTIAL_NO_RELEASE`: this does not close global
reasoning/protocol, Laguna long-context, media, full suite/build, or
Sequoia/Tahoe signing/notarization gates.

## Failure reproduced before the fix

With `gateway_single_model_mode=true`, the app had:

- `jangq-ai/Laguna-S-2.1-JANG_2L` running on port `8008`.
- A separate `dealignai/LFM2.5-8B-A1B-MXFP4-CRACK` engine still alive on port
  `8051`.
- The LFM persisted session row was `stopped` and pointed at an older saved
  port, so the old DB-status-only single-model path did not stop the live
  process.

This is the exact release-blocking behavior called out in
`20260722_laguna_mixed_swa_l2_gibberish/README.md`.

## Source fix

Files:

- `panel/src/main/sessions.ts`
  - manual Start now calls `stopDetectedLocalEnginesForSingleModel()` before
    launching a target session;
  - healthy already-running target processes are adopted instead of duplicated;
  - startup `detectAndAdoptAll()` prunes multiple healthy detected processes
    when single-model mode is on, keeping the active/newest candidate and
    terminating the rest;
  - gateway code has a public
    `enforceSingleModelLocalProcessContract()` entry point for the same local
    process contract.
- `panel/tests/session-single-model-start.test.ts`
  - pins manual Start, stale detected process termination, startup adoption
    pruning, live-PID termination, and target adoption contracts.

## Tests

Remote checkout:

`/Users/eric/mlx/vllm-mlx-release-1.6.13`

Commands:

```sh
cd panel
npm test -- --run tests/session-single-model-start.test.ts tests/api-gateway-single-model.behavior.test.ts
npm run typecheck
```

Result:

- `tests/session-single-model-start.test.ts`: `5 passed`
- `tests/api-gateway-single-model.behavior.test.ts`: `29 passed`
- `tsc --noEmit`: passed

## Live Electron evidence

Relaunch used the current Electron dev app on CDP `127.0.0.1:9335` and user data
dir `/Users/eric/.vmlx-v1613-responsive-dev`.

Evidence artifacts:

- `relaunch-log-excerpt.txt`
  - includes `Using vMLX userData override: /Users/eric/.vmlx-v1613-responsive-dev`;
  - includes `DevTools listening on ws://127.0.0.1:9335/...`;
  - includes `Adopted 1 vmlx-engine process(es)`;
  - includes `jangq-ai/Laguna-S-2.1-JANG_2L on port 8008`.
- `processes-after-relaunch.txt`
  - contains the Electron dev process and exactly one `vmlx-engine serve`
    process: Laguna PID `62411` on port `8008`;
  - contains no live LFM `vmlx-engine serve` process.
- `sessions-after-relaunch.json`
  - Laguna row is `status="running"`, `pid=62411`, `port=8008`;
  - LFM row is `status="stopped"`, `pid=null`, with an updated
    `last_stopped_at`.

## Remaining release boundaries

Still open after this scoped fix:

- global reasoning/content/tool rail correctness across Chat, Responses,
  Anthropic, Ollama, and Electron;
- Laguna strict reasoning/content behavior and protocol breadth;
- SSD-only partial-prefix L2 proof across more architecture classes;
- gateway LAN/port soak beyond this orphan-prune row;
- full test/build, version bump, Sequoia/Tahoe DMG signing/notarization, install
  smoke, release upload, and manifest publishing.

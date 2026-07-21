# Effective generation defaults across bundle and session startup policy

Date: 2026-07-20
Host: `erics-m5-max.local`
Repository: `/Users/eric/mlx/vllm-mlx-release-1.6.13`
Branch: `codex/postrelease-ui-drawers-20260720`
Base source: `3916a6cf075d4f666f81a90795b70096bfdece36`

## Defect

Chat Settings displayed bundle metadata only. That was correct for ordinary
sessions, but false for a native-MTP session launched in the default
`deterministic` mode. The panel passes
`--native-mtp-sampling-policy deterministic-defaults`; the server then installs
greedy omitted-request defaults (`temperature=0`, `top_p=1`, `top_k=0`,
`min_p=0`) so the identity-verified native-MTP path is exercised. The Qwen
drawer instead showed its stochastic bundle defaults (`1 / .95 / 20 / 0`).

This was a UI/effective-policy parity defect, not a Qwen quantization or model
artifact defect.

## Source repair

- `panel/src/shared/effectiveGenerationDefaults.ts` is the single policy
  helper. It applies the exact deterministic native-MTP startup defaults only
  when model detection says native MTP is supported and the stored session mode
  is `deterministic` (including older sessions that omitted the mode field).
- `ChatSettings.tsx` applies that helper after bundle and model detection.
- Both `ChatModeToolbar.tsx` and `SessionView.tsx` now pass the real serialized
  session config into Chat Settings.
- `panel/src/env.d.ts` now exposes the already-returned `nativeMtp` detection
  contract instead of forcing renderer code to guess or cast.
- Explicit per-chat sampling values retain request precedence. `auto` and `off`
  native-MTP modes continue to show bundle defaults.

## Automated validation

- `tests/effective-generation-defaults.test.ts`: 5 passed.
- `tests/settings-flow.test.ts`: 287 passed.
- `tests/request-builder.test.ts`: 74 passed.
- `tests/chat-settings-reset-policy.test.ts`: 4 passed.
- `tests/layout-shell.test.ts`: 191 passed.
- Combined focused run: 5 files and 561 tests passed.
- Panel TypeScript typecheck: passed.

## Live Electron and raw API proof

| Gate | Result | Evidence |
|---|---|---|
| Laguna JANGTQ ordinary bundle inheritance | **PASS-LIVE scoped** | Real stopped-session drawer visibly showed `.70 / .90 / top-k Off / min-p Off / repetition 1.0 / max 2048`; real Start replaced DSV4 and eagerly loaded Laguna before a request. The Electron turn exact-finaled `LAGUNA-SAMPLING-INHERIT-DONE`. Session logs resolved `temperature=.7, top_p=.9, max_tokens=2048`. |
| MiniMax-M3 native typed ordinary bundle inheritance | **PASS-LIVE scoped** | Drawer visibly showed `1 / .95 / Off / Off / 1`. Real Start replaced Laguna and eagerly loaded M3. Electron exact-finaled `M3-SAMPLING-INHERIT-DONE`; 128 tokens restored as `paged+disk`. Logs resolved `temperature=1, top_p=.95`. |
| Qwen MXFP/MTP pre-fix mismatch | **FAIL reproduced** | `q27-mxfp-sampling-defaults-current.png` shows bundle `1 / .95 / 20`, while `q27-mxfp-session-logs-current.json` records the effective omitted request as `0 / 1 / top-k omitted`. |
| Qwen deterministic native-MTP inherited UI/runtime | **PASS-LIVE scoped** | Patched drawer visibly shows `0 / 1 / top-k Off / min-p Off / repetition 1`. Same-chat Electron exact-finaled `Q27-MXFP-SAMPLING-EFFECTIVE-POSTFIX`; SQLite records non-empty content, no reasoning/warning, and `paged+ssm+tq-native`. Runtime logs resolve `0 / 1`. |
| Explicit per-chat stochastic override | **PASS-LIVE scoped** | The real drawer was changed to `1 / .95 / 20 / 0 / 1` and saved. Electron emitted seven distinct progressive turn states, exact-finaled, and runtime logs resolved `temperature=1, top_p=.95, top_k=20`. Real Reset returned the drawer to effective inherited `0 / 1 / Off / Off / 1`. |
| Raw Responses omitted versus explicit | **PASS-LIVE scoped** | `q27-sampling-raw-ab.json`: omitted produced 8 progressive content deltas and one completed terminal; explicit produced 10 deltas and one completed terminal. Both exact-finaled. Runtime logs independently resolve omitted `0 / 1` and explicit `1 / .95 / 20`. |
| Native MTP remained active | **PASS-LIVE scoped** | `q27-health-sampling-after-raw.json` reports native MTP active at depth 3, `deterministic-defaults`, text+VL, and the last explicit stochastic request used rejection-sampling MTP with 18 drafted / 6 accepted tokens including depth-2 and depth-3 accepts. |
| One-model ownership during swaps | **PASS-LIVE scoped** | DSV4 -> Laguna -> M3 -> Qwen were each launched through the real UI. `sampling-effective-engine-processes.txt` contains exactly one remaining engine, Qwen PID 45354, with deterministic native-MTP argv. |
| Clean renderer/process relaunch | **PASS-LIVE scoped** | The old dev Electron and engine process tree was terminated. A new renderer was launched with the persisted profile and CDP 9335; `fresh-relaunch-engine-start.txt` records `[Engine Manager] Found in PATH: /Users/eric/mlx/vllm-mlx/.venv/bin/vmlx-engine`. Qwen was then started again through the real Start button and reached healthy eager state with `last_request_time=null`. `q27-sampling-postrelaunch-effective-drawer.png` visibly shows the effective `temperature=0` and `top_p=1` values from the fresh renderer. |
| Clean-relaunch output emission | **PASS-LIVE scoped** | `q27-mxfp-sampling-postrelaunch-stream.json` records 113 distinct observed UI states before terminal completion. `q27-mxfp-sampling-postrelaunch-final.png` shows a separate 1211-character reasoning rail and exact non-empty visible final `Q27-MXFP-SAMPLING-POSTRELAUNCH-DONE`; `fresh-relaunch-assistant-row.json` preserves the separate database fields and terminal metrics. |

## Honest boundary

This closes the native-MTP effective-default mismatch and provides ordinary
bundle-inheritance spot checks for JANGTQ Laguna and typed M3. It does not yet
prove every model family, a non-neutral repetition-penalty bundle, Chat/
Anthropic/Ollama sampling overrides, signed-app repetition, or all session
config combinations. Those remain in the master matrix.

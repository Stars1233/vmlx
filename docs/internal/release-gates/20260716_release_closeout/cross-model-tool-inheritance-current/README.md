# Cross-model new-chat tool-setting inheritance proof

Status: `PASS-LIVE` for the visible last-chat inheritance contract, SQLite
persistence, and an inherited real tool call. Generation, sampler, prompt, and
reasoning settings remain model-derived.

## Reproduction and source trace

- Before the repair, built-in tools and `/Users/eric/mlx/vllm-mlx` were saved
  on a Qwen chat. After the visible single-model switch to Bonsai, a fresh chat
  showed **Enable Built-in Coding Tools** unchecked and no working directory.
  The earlier Qwen failure row consequently narrated a simulated tool instead
  of receiving functions.
- The owning `chat:create` path searched only same-model siblings and selected
  the newest sibling before checking whether it had an override row. This both
  contradicted the UI's “last chat” wording and stopped inheritance at an empty
  sibling.
- Commit `d9cef0b0c` scans recent chats in update order, excludes the newly
  inserted chat, skips rows without overrides, and uses the newest actual
  override. `buildNewChatInheritedOverrides` still copies only its explicit
  tool/workspace allow-list; it does not inherit temperature, top-p/k/min-p,
  output/thinking caps, prompt, or reasoning mode. A starred default profile
  remains higher priority.
- `source-trace.txt` records the current IPC and allow-list implementation.
  `affected-tests.txt` records 299/299 affected panel tests and typecheck.

## Live cross-model proof

- The Electron main process was fully restarted on the edited source. Bonsai
  was visibly loaded, its tool setting and working directory saved, then the
  UI switched to Qwen 27 MTP. A fresh Qwen chat inherited built-in tools, the
  allowed category toggles, and `/Users/eric/mlx/vllm-mlx` without manual edits.
- `q27-cross-model-tool-inheritance-db.json` records the new chat override:
  `builtin_tools_enabled=1`, the inherited working directory and categories,
  while sampling, prompt, output/thinking limits, and `enable_thinking` remain
  SQL NULL. The main log names Bonsai chat `9f39fb3c` as the inheritance source.
- `q27-cross-model-tool-inheritance-bottom.png` visibly shows the current Qwen
  model/PID, checked inherited tool categories, real `Info README.md` card, and
  exact final answer.
- `q27-cross-model-tool-inheritance-result.json` proves the inherited state was
  operational: 298 reasoning paints, 10 progressive content paints, exactly
  one real `file_info(README.md)`, a matching result, and exact
  `Q27-INHERIT-TOOL1-DONE`. The 28-second native reasoning rail remains a
  separate latency/reliability concern, not a settings inheritance failure.

## Release boundary

This closes cross-model last-chat inheritance for tool/workspace ergonomics on
the current Electron build. Profile priority, reset behavior, and broader
settings round-trips remain in the aggregate settings matrix.

# Qwen terminal and exact-once stream proof

Status: `PASS-LIVE` for the scoped shared terminal contract and explicit Qwen
exact-once request; broader parser/model reliability remains `PARTIAL`.

Source head: `9618e2e46`.

Source ownership:

- `vmlx_engine/server.py:1065-1098`: hide an internal reasoning-pass terminal
  only while a real visible-answer continuation remains pending.
- `vmlx_engine/server.py:5483-5487`: explicit Qwen exact-once requests use zero
  post-call grace; ordinary multi-call requests keep the generic grace.
- `vmlx_engine/server.py:16719-17545,18378-19193`: preserve the complete
  early-stopped candidate for Chat and Responses final parsing.

Failure evidence retained:

- `q27-admission-stream1.json`: pre-fix Chat exposed `["length", "stop"]`.
- `q27-terminal-electron1.{json,png}`: pre-fix exact-once Electron request
  executed two byte-identical `file_info(panel/package.json)` calls.
- `q27-tool-warm1.json` and `q27-tool-bypass1.json`: cache-on and cache-bypass
  raw controls each generated one call, falsifying TQ/SSM prefix corruption as
  the cause of that duplicate.

Passing evidence:

- `q27-terminal1.json`: post-fix Chat turn 1 and turn 2 each expose only one
  final `stop`; Responses exposes one completed terminal. Reasoning and content
  remain separate progressive deltas.
- `q27-exactonce-electron2.{json,png}`: live Electron emits 279 reasoning and
  ten content events, executes one `file_info(panel/package.json)`, persists
  one matching call/result, restores 128 `paged+ssm+disk` tokens, and returns
  exact `Q27-EXACTONCE-ELECTRON2-DONE`.
- `focused-tests.txt`: current pushed head passes 131 selected terminal,
  answer-pass, and server tests; three tests are intentionally deselected by
  repository configuration.

No output deduplication or synthetic answer text was added. The fix constrains
the server-owned stream contract and preserves model output at the first
schema-valid explicit exact-once call boundary.

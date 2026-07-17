# Terminal cache-admission proof

Status: `PASS-LIVE` for the shared admission invariant; overall release remains
`PARTIAL_NO_RELEASE`.

Source head: `016d661ca` (followed by stream/parser head `9618e2e46`).

Source ownership:

- `vmlx_engine/engine_core.py:100-104,189-195,292-305,472-475`
- `vmlx_engine/mllm_scheduler.py:882-886,4026-4053,4074-4079,4180-4184`

Both async schedulers dispatch terminal output before slow cache persistence,
but close request admission until paged/TQ/typed companion cleanup finishes.
The gate reopens in `finally` and on cancellation/error.

Live evidence:

- `mm27-admission1.json`: cold then immediate identical RAM request; the second
  request selects all 3,752 reusable tokens as `paged+tq-native`.
- `mm27-admission1-restart.json`: process-restart restore of all 3,752 tokens as
  `paged+disk+tq-native`, followed by a full resident hit.
- `mm27-admission1-logs.png`: visible Electron Logs proof that the full store
  completed before the next prefix selection.
- `mm27-admission-stream1.json`: Chat two-turn and Responses progressive stream
  control. Two leading newlines are retained as a strict-format miss.
- `mm27-admission-electron1.{json,png}`: live Electron one-tool row, 63
  reasoning events, ten content events, one `file_info`, exact final.
- `q27-admission1.json`: cold then immediate identical RAM request; the second
  request selects all 4,622 reusable tokens as `paged+ssm`.
- `q27-admission1-restart.json`: full 4,622-token `paged+ssm+disk` restart
  restore followed by a resident hit.
- `q27-admission1-health-after-restart.json`: q4 native-TQ block hits, block
  disk hits, one native SSM companion disk hit, and no unsafe KV-only reuse.

The evidence proves the shared admission ordering on one plain full-attention
TQ4 model and one hybrid q4-KV plus native-SSM model. It does not replace the
remaining architecture-specific matrix.

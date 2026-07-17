# DSV4 eager session materialization evidence

Scope: DSV4 only, commit `1e15c94bd`, 2026-07-17.

Source ownership is `vmlx_engine/utils/tokenizer.py:1123-1134`: the DSV4
JANGTQ loader evaluates stored quantized parameters during session start with
`skip_params_eval=False`. This does not run a synthetic prompt or populate a
prefix cache.

The real Electron dev build was used to stop and start the DSV4 session. The
post-start, pre-prompt health capture records:

- `model_loaded=true`
- `last_request_time=null`
- `memory.active_mb=99724.7`
- no running/waiting scheduler request and zero cached prompt tokens

The before/after screenshots, process capture, and health JSON are retained in
this directory. Focused DSV4 materialization tests passed 18/18. This is not
evidence that every architecture route eagerly materializes.

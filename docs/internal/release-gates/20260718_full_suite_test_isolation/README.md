# Current-source full Python suite checkpoint

Date: 2026-07-18

Scope: post-release test isolation and stale source-contract repair on branch
`reconcile/1.5.68`, starting from `38d113ccd`.

## Source trace

- `tests/test_turboquant_cache_contract.py` now restores the serve-time
  TurboQuant/SSM policy environment after every in-process CLI test. The CLI
  intentionally mutates process environment for a real server lifetime, but
  these tests stop at `uvicorn.run` and continue in the same pytest process.
  Before this fixture, `test_explicit_kv_quantization_disables_loader_turboquant`
  left `VMLX_DISABLE_TQ_KV=1`, causing later disk-cache tests to fail by test
  order rather than runtime behavior.
- `tests/test_engine_audit.py` now checks the behavior of
  `_auto_thinking_partition_allowed` and verifies that both streaming Chat and
  Responses endpoints invoke the shared helper. It no longer matches the
  removed pre-refactor inline conditional.
- `tests/test_streaming_reasoning.py` now pins the current progressive M3
  visible-answer stream (`_stream_with_keepalive` plus
  `response.output_text.delta`) instead of an obsolete log message.

## Verification

- Focused shared-policy/source contracts: `2 passed, 715 deselected`.
- TurboQuant ordering reproduction after the isolation fixture: the whole
  `test_turboquant_cache_contract.py` file followed by the two formerly
  failing disk-cache tests passed `10/10`.
- Bundled Python was rebuilt from current vMLX source and clean JANG source
  `/Users/eric/.cache/vmlx-release/jang-clean-9081c924/jang-tools`; the bundled
  verifier passed version, critical source hashes, relocatable shebangs, and
  all listed runtime imports.
- Full Python suite with Node available in `PATH`:
  `5942 passed, 96 skipped, 261 deselected, 2 warnings in 246.95s`.

Evidence:

- `full-pytest.log`
- `bundled-python-verify.log`

## Boundary

This checkpoint is source/test evidence only. No model was loaded and no chat
generation was run for this checkpoint, so Electron model behavior, raw API
streaming, cache reuse, and release readiness remain outside this proof.

The generated current no-heavy regression orchestrator completed `status=open`.
Several legacy proof runners still reject renamed tests/current UI source
markers even though their child pytest/vitest commands pass. Those meta-audit
contracts must be reconciled independently; this checkpoint does not relabel
them green.

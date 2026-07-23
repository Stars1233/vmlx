# v1.6.17 release-head suite audit

Status: `SOURCE SUITE PASS / BUNDLE DRIFT OPEN`.

Source sweep at `493f418d3` established:

- panel suite: `2490 passed, 3 skipped`;
- panel TypeScript: pass;
- Electron production renderer build: pass;
- Python collection initially produced false async failures because the remote
  development venv lacked the declared `pytest-asyncio` test dependency;
- after installing the test-only dependency, the failed set reduced to seven
  concrete rows.

Six rows were corrected and passed together on the M5 Max (`6 passed in
3.65s`; see `focused-fixes.txt`):

1. MiniCPM-V-4.6 was remapped to the MiniCPM-o runtime while retaining an
   incompatible pre-existing mlx-vlm prompt format. Runtime and prompt aliases
   are now updated atomically.
2. Responses incremental usage during a bounded visible-answer pass omitted
   the first pass's `cached_tokens`/`cache_detail`. The continuation now
   preserves that request-level cache accounting.
3. A real ZAYA1-8B-MXFP4 bundle uses legacy `rope_scaling:false`, which
   Transformers 5 rejects for tokenizer config even though it means no
   scaling. ZAYA now normalizes only `false` to `None` in memory and passes a
   version-stable generic `PreTrainedConfig` to AutoTokenizer; the official
   bundle is not changed. The generic path is required because the source
   development environment has Transformers 5.7 without a model-specific ZAYA
   config module, while the M5 Max test environment has Transformers 5.14.
4. Command-preview parser coverage was stale after canonicalization moved into
   the shared resolver.
5. The low-memory preflight test still required arbitrary `killByPort`; the
   product correctly uses exact-session/model owned-port cleanup and refuses
   an unowned process.
6. The rotating-cache source-shape assertion had not followed the metadata
   bounds logic into its shared helper.

The seventh failure is an intentional packaging guard:
`verify-bundled-python.sh` detects source/bundled-engine SHA drift. It must
remain open until the final versioned source is frozen and
`bundle-python.sh` rebuilds the bundled runtime. No stale bundle was accepted
to make the suite green.

The complete source suite at `b05dc6840`, with only the intentional
bundled-Python integrity row deselected, passed on the M5 Max:
`6406 passed, 96 skipped, 93 deselected` in 255.41 seconds. The ZAYA
version-neutral follow-up passed its normalization and real-bundle focused
tests (`2 passed`); the complete suite must now be rerun at that follow-up
head.

This checkpoint is not yet release-ready. Required next evidence is the final
head Python rerun, rebuilt-bundle verification, raw live API protocol smoke,
and signed/notarized installed-app smoke.

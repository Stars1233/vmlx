# vMLX real-implementation + cleanup campaign — final graded report

Date: 2026-07-10

Branch: `reconcile/1.5.68`

Base: `origin/main` at `0d4ac2b0e` (fast-forward pull reported current)

Commit: this report is committed with the implementation; the resulting SHA is printed in the handoff.

## Result

This campaign closes the implementation/truthfulness defects for TurboQuant live encode wiring, `/v1/capabilities`, request-local seed, Nemotron selective attention TQ, and Laguna mixed-SWA reachability. Gemma and Hy3 reasoning parity are live green. MiniMax native reasoning and real UI click-through remain honestly open; no hidden sampling override, prompt-only repair, output repair, package, signing, or production-app mutation was used.

## #78 — TurboQuant live encode

`TurboQuantConfig.compress_after` is now a real engine-owned field. It is threaded through `jang_config.turboquant`, loader auto-config, hybrid/plain cache factories, memory-cache clone/reconstruction, MLLM reconstruction, runtime telemetry, logs, health, cache stats, and capabilities.

The real `TurboQuantKVCache.compress()` boundary records requested/compressed tokens plus resident bytes before and after. Capabilities now distinguish objects from encode and say exactly: objects active; live encode disabled/enabled; stored prefix q4. The prior 3-bit/5x resident-memory claim was removed because the live implementation retains packed and decoded/joined buffers.

| Family / component | Encode armed | Greedy | temp-0.9 / reasoning | Cache proof | Resident result | Default |
|---|---:|---|---|---|---:|---|
| Hy3 plain KV, 80 TQ slots | PASS, 64 tokens/layer | exact cold/warm | seeded text exact; reasoning matrix 10/12 while armed | 748-token `memory+tq`; codec counters advanced | last codec transition -1,310,400 bytes, but coherence gate failed | OFF |
| Qwen3.6 attention KV, 16 TQ + 48 native SSM | PASS, attention only | exact cold/warm | temp-0.9 exact | 1,233-token `paged+ssm`; SSM kept native | +1,637,280 bytes | OFF |
| Gemma4 full attention, 8 TQ + 40 rotating SWA | PASS after global-head-dim fix | exact cold/warm | temp-0.9 exact | 1,596-token `memory` hit | +2,298,176 bytes aggregate | OFF / explicit SWA opt-in only |

Grade: implementation PASS; default-on gate FAIL for every tested family. This is the intended safe result: live encode is usable for explicit diagnostics, but no family is enabled by default without both coherence and memory wins.

## Reasoning parity

Concrete `reasoning_effort` now implies `enable_thinking` only when it is unspecified and the registry says `supports_thinking=True`. Explicit off, non-thinking-family guards, family normalizers, Mistral restrictions, and MiniMax custom off remain authoritative. The mapping is applied across OpenAI Chat/Responses, Anthropic, and Ollama routes.

| Family | OpenAI | Ollama | Anthropic | Grade |
|---|---:|---:|---:|---|
| Gemma4 | 4/4 | 4/4 | 4/4 | PASS 12/12 |
| Hy3, encode OFF | 4/4 | 4/4 | 4/4 | PASS 12/12 |
| MiniMax-M2.7 full artifact | 1/4 | 4/4 | 3/4 | FAIL 8/12 |

Gemma's formerly failing OpenAI effort cell produced 1,158 reasoning characters, the correct 90 km/h answer, and no raw leak. Hy3's encode-off 12/12 versus encode-on 10/12 is also the decisive #78 coherence rejection.

MiniMax classification: the 36 GB Small artifact loops through 4,096 native-reasoning tokens without a final; the full artifact is better but OpenAI greedy native on/auto/effort still exhausts 900 tokens without final. The supplied `/tmp/reasoning_parity.py` Anthropic off cell sends no disabled flag and is byte-identical to auto, so that cell cannot express off. MiniMax stays OPEN; custom reasoning off itself is live correct. No hidden temperature change or output synthesis was added.

## Cache policy and remaining ledger items

- `/v1/capabilities`: PASS, live 200 response for the active model.
- Seed: PASS on fresh-cache `/v1/completions`; same seed byte-identical, different seed diverged. Chat SSE emitted six visible deltas ending `STREAM-SEED-OK`. Seed is request-local and also drives native-MTP rejection/residual sampling.
- Memory q4 replay: PASS. Scheduler-owned memory-prefix q4 tuples are stored as stream-independent packed NumPy buffers and converted on the owner worker before dequantization. This removed the reproduced MiniMax `no Stream(gpu, 0)` empty-200/process-abort failure. Paged cache retains MLX typed blocks.
- Nemotron-Omni allow-list: PASS. Live layout is 6 attention TQ plus 23 native SSM/conv slots; 709-token cold/warm exact, 702 cached tokens, `paged+ssm+tq`, SSM companion entry present.
- Laguna SWA reachability: PASS. Dedicated loader reaches 10 full-attention TQ plus 30 rotating slots; 602-token cold/warm exact, 601 cached tokens, `memory+tq`.
- Paged default: source/type contract PASS: new sessions keep prefix on and paged/block-disk off; structural hybrid families still force the typed paged lane.
- UI click-through and paged toggle: FAIL/unproven. The mandatory in-app browser connector rejected required sandbox metadata before connecting. No alternate browser backend was substituted.
- MTP UI: source/typecheck PASS, live visual FAIL/unproven. Hy3 is recognized, its settings include depth/source/scope/native cache type, and the renderer shows native type post-load data. Browser verification remains open.

## Dead/zombie/duplicate cleanup

Static AST name/reference inventory was followed by exact project grep. Removed only helpers with no production or test references:

- `server._nearly_equal`
- `server._bundle_index_has_tensor_prefix`
- `Scheduler._decode_tokens`
- `Scheduler._validate_single_cache`
- `native_mtp._vision_weight_keys`
- obsolete nested `_render_dsml_examples`

Kept intentionally: import hooks, decorators/Pydantic validators, parser registries, monkey-patch entry points, test-addressed compatibility methods, M3/DSV4/MLA cache shims, and the measured bundle-stamp native-MTP block. Static name counts are not sufficient evidence to remove those dynamic surfaces. The legacy Hy3 JANG_2K profile-name MTP block has no sibling; the remaining comment documents why measured bundle stamps replaced it.

## Verification

- Focused Python contracts: 333 PASS / 1 known pre-existing reasoning-display failure before final additions; campaign/cache/reasoning subset subsequently 18/18 PASS.
- Panel TypeScript: PASS (`tsc --noEmit`).
- Panel focused tests: 187 PASS; two path failures were invocation-CWD artifacts. Running from panel cwd is blocked by Electron 28/Vitest 4 ESM incompatibility; source typecheck remains green.
- Full pytest baseline: 54 failed, 5,775 passed, 94 skipped, 92 deselected.
- Full pytest final: 54 failed, 5,786 passed, 94 skipped, 92 deselected. Failure count is identical to baseline; the 11 campaign tests are all new passes. Zero NEW failures.
- No model server remains running after proof.
- Models used only from `/Volumes/EricsLLMDrive/`.
- `/Applications/vMLX.app` was not touched.
- Packaging/signing/notarization commands were not run.

## Open boundary

Do not mark the overall campaign fully closed: MiniMax OpenAI greedy native reasoning is still red; the real UI/PAGED-toggle/MTP rendering proof is blocked; #45 q4 stochastic cache divergence remains an explicit per-family numerical gate. All newly implemented capability claims above have live output, logs, stream data, and cache telemetry.

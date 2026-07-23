# OpenPangu current-source Electron, protocol, and native prompt-L2 gate

Date: 2026-07-23

Status: `VERIFIED-LIVE-SCOPED / OVERALL PARTIAL / NOT RELEASE-READY`

## Exact source and runtime provenance

- Source branch: `codex/v1.6.17-consolidation-20260723`.
- Cache repair and live-proof head:
  `568c1e105d3b017414641bc4738dfc15e8eb2a41`.
- Checkout on both machines:
  `/Users/eric/mlx/vllm-mlx-r17-consolidation`.
- Real isolated Electron app: CDP `9335`, gateway `8088`.
- Exact bundle:
  `/Volumes/EricsLLMDrive/JANGQ-AI/openPangu-2.0-Flash-JANG_3M`.
- Direct model port: `8007`.
- The real Electron Start button launched PIDs `96166`, `96760`, and `97021`
  across the deliberate cache restart checks. The restart argv artifacts retain
  the exact source-module invocation and flags.
- The development app log reports:
  `[Engine Manager] Found development project venv:
  /Users/eric/mlx/vllm-mlx-r17-consolidation/.venv/bin/python3
  (vmlx_engine 1.6.16)`.
  It does not report the installed-app `Found in PATH` wording, so this is
  source-app provenance rather than signed-installed-app proof.

## Bundle-grounded architecture truth

`r17-openpangu-capabilities.json` records:

- family `openpangu_v2`;
- affine `JANG_3M`, group size `128`, configured 3-bit / measured 3.83-bit;
- MLX affine quantized matmul on an Apple M5 Max;
- text-only runtime modality;
- effective parsers `openpangu` tools and `deepseek_r1` reasoning;
- sampling defaults temperature `1.0`, top-p `0.8`, top-k `151552`, min-p
  `0.0`, and max output `16384`;
- typed cache schema `openpangu_v2_composite_v2` with MLA latent KV, DSA
  indexer state, rotating SWA, and path-dependent convolution state;
- generic paged blocks and generic TurboQuant KV disabled for this
  architecture;
- exact full-precision typed prompt-disk L2 with exact or forward-prefix reuse.

The config and JANG sidecar declare three MTP layers, but the bundle index
contains zero MTP tensors and the current runtime reports
`weights_present_runtime_unwired`. MTP is not counted as active.

## Electron three-turn proof

The same Electron-started session completed three inspected UI turns:

1. separate private reasoning plus a two-line visible math answer;
2. one real built-in `file_info(panel/package.json)` call, real `5.2 KB`
   result, and final continuation;
3. history recall of the real size with fresh private reasoning and a
   non-empty visible final answer.

The UI rendered `\(47 \times 19 = 893 < 920 = 46 \times 20\)` as
`47×19=893<920=46×20`. It did not expose raw dollar delimiters, raw backslash
commands, reasoning tags, or tool markup. Retained screenshots:

- `r17-openpangu-ui-turn1.png`
- `r17-openpangu-ui-turn2-final.png`
- `r17-openpangu-ui-turn3.png`
- `cache/r17-openpangu-cache-exact-ui.png`

## Direct and gateway protocol proof

The retained agentic harness artifacts cover direct port `8007` and Electron
gateway `8088` for Chat Completions, Responses, Anthropic Messages, and
Ollama. Across the retained stream/non-stream flows:

- private reasoning stayed separate from visible content;
- tool rounds emitted no visible prose;
- `file_info(panel/package.json)` and `run_command(pwd)` were emitted exactly,
  executed by the harness, and continued from their real results;
- final text was progressive, terminals were truthful, and timestamps were
  monotonic;
- raw control markup did not leak and reasoning was neither duplicated into
  content nor stale across distinct turns.

The raw direct and gateway LaTeX/currency captures are byte-faithful. KaTeX is
an Electron display concern only; API payload bytes are not rewritten.

The harness did not run abort/recovery injection in this row. That broader
gateway lifecycle boundary remains open.

## Global cache discriminator defect and repair

### Root cause

Normal generation requests carry `_cache_extra_keys` so a prompt rendered under
one generation-template contract cannot collide with another. The scheduler
previously rejected memory-aware prefix, legacy prefix, and prompt-disk L2
lookup whenever those keys were present. Ordinary Chat and Responses requests
therefore missed every cache backend even when their token prefix and
discriminator were identical.

### Structural correction

Commit `568c1e105d3b` preserves the discriminator in the cache identity instead
of dropping it or disabling cache reuse:

- `vmlx_engine/cache_key.py` canonicalizes the marker;
- `MemoryAwarePrefixCache` and `PrefixCacheManager` scope entries by it while
  preserving the legacy unmarked key shape;
- `DiskCacheManager` hashes and indexes token prefixes with the marker, migrates
  the SQLite schema, and filters longest-prefix candidates by marker;
- the scheduler passes the marker to each backend while retaining old call
  shapes for callers with no discriminator.

Current-head regression artifact:
`cache/r17-cache-discriminator-current-head-tests.log`:

- `152 passed in 13.94s`;
- includes discriminator backends, cache bypass, generation-prompt key, disk,
  memory, and prefix-trie suites.

## Native prompt-L2 live proof with Paged RAM off

Every live engine argv retained `--no-paged-cache --enable-disk-cache
--disk-cache-max-gb 10`.

### Cold store and resident exact hit

- Cold Chat request: one miss, one memory entry, one prompt-L2 entry, and
  `1,947` prompt tokens stored.
- Same-process exact request: one memory hit and `1,946` cached tokens.
- Both streams progressively emitted `PRIME-DONE` and terminated with
  `finish_reason:"stop"` plus `[DONE]`.

### Exact SSD restore after real Electron restart

- The real UI stopped PID `96166` and started PID `96760`.
- Before replay: RAM entries `0`; the one `1,947`-token SSD entry remained.
- Identical Chat replay: disk hits `1`, stores `0`, scheduler detail `disk`,
  and `1,946` cached tokens.
- The replay again emitted `PRIME-DONE` and one truthful terminal.

### Changed-tail partial SSD restore after another real restart

- A raw-completions cold pass stored a `3,544`-token common prefix.
- The real UI stopped PID `96760` and started PID `97021`.
- Before replay: RAM entries `0`; both SSD entries remained.
- The extended prompt had `3,565` input tokens.
- SSD restored `3,543` cached tokens and prefetched only the `22`-token new
  suffix.
- SSE usage reports
  `prompt_tokens_details.cached_tokens=3543` and
  `cache_detail="disk"`.
- The visible stream progressively completed `RAW-EXTENDED-DONE` and ended
  with one `[DONE]`.

This is real partial-prefix reuse of OpenPangu's typed prompt snapshot. It is
not generic block reuse: health truthfully reports `blocks:0`, generic paged
blocks unsupported, and block-L2 tokens `0`. Arbitrary suffix-only matching is
not a valid transformer cache operation. The safe contract proven here is a
shared prefix restored from SSD plus prefill of the unmatched suffix.

## Evidence

Protocol and UI:

- `openpangu-chat-agentic-live.json`
- `openpangu-responses-agentic-live.json`
- `openpangu-anthropic-agentic-live.json`
- `openpangu-ollama-agentic-live.json`
- `r17-openpangu-capabilities.json`
- `r17-openpangu-health.json`
- `r17-openpangu-latex-direct-v2.sse`
- `r17-openpangu-latex-gateway-v2.sse`

Cache:

- `cache/r17-openpangu-cache-health-after-cold.json`
- `cache/r17-openpangu-cache-health-after-exact.json`
- `cache/r17-openpangu-cache-health-after-restart-before-replay.json`
- `cache/r17-openpangu-cache-health-after-restart-disk.json`
- `cache/r17-openpangu-cache-health-prefix-restart-before.json`
- `cache/r17-openpangu-cache-health-after-raw-extended-disk-partial.json`
- `cache/r17-openpangu-cache-chat-cold.sse`
- `cache/r17-openpangu-cache-chat-exact.sse`
- `cache/r17-openpangu-cache-chat-restart-disk.sse`
- `cache/r17-openpangu-cache-raw-extended-disk-partial.sse`
- `cache/r17-openpangu-cache-restart-argv.txt`
- `cache/r17-openpangu-cache-prefix-restart-argv.txt`
- `cache/SHA256SUMS.txt`

## Remaining boundary

- Generic Paged RAM, block-disk L2, and q4 TurboQuant KV are architecture
  incompatible for OpenPangu and were not enabled.
- MTP is declared but not present/active in this exact artifact/runtime.
- Abort/disconnect/fault recovery, corrupt typed-companion fallback, capacity
  eviction/refault, 512K context, signed-app repetition, full suites, packaging,
  notarization, and publishing remain open.
- Overall v1.6.17 status remains
  `ACTIVE / PARTIAL / NOT RELEASE-READY`.

# MiniMax M3 Electron tool and math checkpoint

Date: 2026-07-23

Status: `FIXED+VERIFIED-LIVE-SCOPED / OVERALL PARTIAL`.

Source head and runtime provenance:

- Source head for the retained renderer/API/cache proof:
  `511fa5e0b14373584dfb978d5cd23222ca7c4b29`.
- Exact bundle:
  `/Volumes/EricsLLMDrive/JANGQ-AI/MiniMax-M3-Coder-Small`.
- Real isolated Electron profile:
  `/Users/eric/.vmlx-r17-consolidation-dev`, CDP `9335`, gateway `8090`.
- The real Sessions-card Start button launched PID `11370` on direct port
  `8008`.
- The app log resolved the project engine at
  `/Users/eric/mlx/vllm-mlx-r17-consolidation/.venv/bin/python3`.
- The launch argv selected both `minimax_m3` reasoning and tool parsers,
  Paged RAM, 64-token blocks, 1,000 maximum blocks, and 10 GB Block Disk
  Cache.

## Defect one: repeated execution of an exactly-once tool

The live M3 turn requested one `file_info(panel/package.json)` call. Before
`b482bec60`, the panel executed three calls because exactly-once retirement
was incorrectly coupled to the separate phrase `reply exactly`.

The correction:

- `requestedOnceToolNames()` treats each explicitly named `exactly once` tool
  as an execution invariant.
- The panel retires that tool schema after its first successful execution.
- A duplicate emitted in the same model pass is rejected before it reaches
  the executor.
- Other tool schemas remain available for genuinely multi-tool agentic work.

Focused remote verification:

- `panel/tests/tool-auto-continue.test.ts`: `23/23` passed.
- Combined tool-auto-continue and chat-metrics selection: `26/26` passed.
- Panel TypeScript typecheck passed.

Live Electron proof after the correction:

- One `file_info` card, one execution, one tool-result continuation, and one
  non-empty final answer.
- The visible result truthfully reported `5.2 KB`.
- Metrics: 89 output tokens, `17.4 tok/s`, 9,418 cumulative prompt tokens,
  5,440 cached tokens, `27.07s` TTFT, and `36.8s` total.

## Defect two: malformed repeated TeX opener

The third M3 turn did not suffer random dollar injection. The raw stored
assistant text contained:

```text
Path: panel/package.json — Size: 5.2 KB
$43 and inline TeX: \(\(47 \times 19 = 893 < 920 = 46 \times 20\)
```

`$43` was explicitly required as literal currency by the user prompt. The
actual malformed portion was the model-emitted duplicate `\(\(` opener with
only one closer. KaTeX correctly rejected that invalid nesting, and the prior
fallback therefore left visibly broken punctuation.

Commit `0592404d8` adds one renderer-resilience rule shared by completed
answers and the actively streaming reasoning rail:

- collapse only immediately adjacent repeated `\(`/`\)`, or `\[`/`\]`
  delimiter tokens;
- leave currency, ordinary parentheses, code spans, and code fences intact;
- keep raw API bytes unchanged.

Focused remote verification:

- `panel/tests/math-markdown.test.ts`: `17/17` passed.
- Panel TypeScript typecheck passed.

Live current-source renderer proof:

- The saved malformed third turn hot-reloaded through the exact current
  renderer.
- DOM inspection found `.math-inline .katex` with readable text
  `47×19=893<920=46×20`.
- DOM inspection found no `.math-fallback`, no raw `\times`, and no raw math
  delimiter.
- The expanded Reasoning rail also rendered the malformed stored expression
  as KaTeX; literal `$43` remained ordinary currency.

## Defect three: escaped Unicode operator inside TeX

The raw direct and gateway protocol probe asked the model to preserve
`\times`. M3 instead emitted the same malformed sequence on every route:

```text
M3-RAW-MATH
CURRENCY=$43 TEX=\(47 \× 19 = 893\)
```

Source inspection confirmed the M3 reasoning parser only partitions
`<mm:think>`/`<think>` markers and the M3 tool parser only strips its own
control envelope. Neither rewrites TeX or Unicode operators. The byte sequence
is model-owned, while the route-to-route equality proves it was not injected
or corrupted by one API adapter.

Commit `f34deae28` adds a presentation-only normalization for a stray slash
before known Unicode math glyphs. It applies inside KaTeX source and the
actively streaming reasoning text; raw API bytes remain unchanged.

Live proof:

- Direct and gateway Chat, Responses, Anthropic, and Ollama each emitted 24
  progressive content deltas with identical raw content and truthful
  terminals.
- All routes preserved `$43`, `\(`/`\)`, and the same `\×`; no route emitted
  KaTeX/HTML.
- The exact-copy diagnostic remains truthfully false because the model emitted
  `\×`, not the requested `\times`.
- A real follow-up Electron turn rendered `47×19=893` as `.math-inline .katex`
  in both answer and reasoning. DOM had no fallback, raw `\×`, or raw
  `\times`.

## Three-turn Electron evidence

1. No-tool math/currency: separate 903-character reasoning rail, 138 visible
   characters, no tool call, `19.5 tok/s`.
2. Exactly-once tool: one real `file_info` execution and one result
   continuation, `17.4 tok/s`.
3. No-tool history recall: exact prior path and size, separate 607-character
   reasoning rail, zero tools, corrected KaTeX rendering, `18.4 tok/s`.

Retained screenshots:

- `m3-ui-turn1-reasoning-math.png`
- `m3-ui-turn2-exact-once-tool.png`
- `m3-ui-turn3-history-math.png`
- `m3-ui-turn3-expanded-reasoning-math.png`
- `r17-m3-ui-escaped-unicode-math-fixed.png`

## Direct and Electron-gateway protocol evidence

- Chat and Responses, direct and gateway:
  - separate progressive reasoning and content deltas;
  - one schema-valid `file_info(panel/package.json)`;
  - one real tool-result continuation;
  - progressive exact final and completed terminal;
  - matching stream and non-stream output.
- Anthropic and Ollama, direct and gateway:
  - separate progressive reasoning;
  - one schema-valid `file_info`;
  - real tool-result continuation;
  - progressive exact final and truthful `message_stop`/`stop`.
- A separate thinking-Off matrix proved all four protocols produce the same
  exact eight-line answer in stream and non-stream modes. No reasoning delta
  appeared while explicitly Off.

## Native MSA/indexer block-L2 hierarchy

The live health schema identifies the exact cache as
`minimax_m3_msa_v1` / `native_msa_sparse_kv`. It retains dense attention KV
for layers 0-2, sparse MSA/indexer state for layers 3-59, and the
`attention_kv`, `msa_idx_keys`, and `absolute_block_index` components.
Generic TurboQuant/storage quantization is correctly disabled for this typed
state; this row does not claim q4 generic-KV storage for M3.

Paged-RAM-on proof:

- Cold request A had 6,944 prompt tokens, reused only the 128-token generic
  template prefix, and reached its first output delta in `11,635 ms`.
- Same-process changed-tail request B reused 6,912/6,944 tokens, prefetched
  only 32 new tokens, and reached its first delta in `1,601 ms`.
- A suffix-only negative placed the same corpus after an unrelated leading
  sequence. It reused only 128/8,030 tokens, proving the implementation did not
  incorrectly splice an arbitrary later suffix into the cache lineage.
- The real Electron Stop button shut PID `11370` down, gateway health reported
  the backend stopped, and port 8008 closed. The real Start button then loaded
  PID `15531` with zero scheduler/L1 tokens and 807 retained SSD blocks.
- Restart changed-tail request D restored 6,912/6,948 tokens as
  `paged+disk`, recorded 108 SSD block hits, prefetched one new block, emitted
  exact `M3-L2-D`, and reached its first delta in `2,053 ms`.

Paged-RAM-off proof:

- In the real Server Settings UI, both `In-Memory Paged Cache (RAM)` and
  `Block Disk Cache (SSD / L2)` were independently enabled. Turning paged RAM
  Off left block L2 checked, enabled, and selectable.
- Save & Restart launched PID `16285` with `--no-paged-cache`,
  `--enable-block-disk-cache`, and health
  `backend_mode=block_disk_only`, `paged_ram_enabled=false`,
  `disk_only=true`, zero scheduler/L1 tokens, and 808 retained SSD blocks.
- Changed-tail request E restored 6,912/6,947 tokens strictly as
  `block-disk`, recorded 108 SSD block hits, prefetched one block, and emitted
  exact `M3-L2-E`.
- A second real UI Stop/Start launched PID `16816` with zero scheduler/L1
  tokens and 809 retained SSD blocks. Changed-tail request F again restored
  6,912/6,950 tokens as `block-disk`, recorded 108 SSD hits, and emitted exact
  `M3-L2-F`.
- The real UI finally restored both paged RAM and block L2. Save & Restart
  loaded PID `17435`; final health is `backend_mode=paged`,
  `paged_ram_enabled=true`, `disk_only=false`, with 810 retained SSD blocks
  and zero pre-request RAM tokens.

This proves safe longest-contiguous-prefix matching, partial suffix prefill,
same-process SSD-only reuse, paged-on SSD promotion, and fresh-process SSD
refault for the exact M3 typed cache. It does not claim arbitrary substring
matching; the suffix-only negative deliberately proves that unsafe behavior is
rejected.

Retained raw artifacts:

- `r17-m3-{direct,gateway}-openai-tools-current.json`
- `r17-m3-{direct,gateway}-anthropic-ollama-tools-current.json`
- `r17-m3-{direct,gateway}-protocol-parity-current.json`
- `r17-m3-{direct,gateway}-raw-math-current.json`
- `m3_raw_math_protocol_probe.py`
- `m3_native_l2_probe.py`
- `r17-m3-l2-paged-on-{a-cold,b-partial,d-restart-disk}.json`
- `r17-m3-l2-paged-off-{e-disk-only,f-restart-disk}.json`
- `r17-m3-l2-suffix-only-negative.json`
- `r17-m3-health-*.json`
- `r17-m3-ui-{cache-controls-expanded-paged-on-disk-on,paged-off-block-disk-still-on,paged-on-after-real-start,paged-off-healthy-before-request,restored-paged-on-block-disk-on}.png`

## Boundary

- The current three-turn Electron text/tool/math/history row is
  `VERIFIED-LIVE-SCOPED`.
- Current direct and gateway Chat/Responses/Anthropic/Ollama streaming,
  reasoning separation, tool continuation, raw-byte, and stream/non-stream
  parity are `VERIFIED-LIVE-SCOPED`.
- Current M3 native sparse/indexer Paged-On partial reuse, restart SSD refault,
  Paged-Off SSD-only partial reuse, and Paged-Off process-restart refault are
  `VERIFIED-LIVE-SCOPED`.
- M3 exact copying of the requested command spelling remains a model-output
  diagnostic: it produced `\×` consistently instead of `\times`. The product
  does not rewrite those raw API bytes.
- M3 disk-cap eviction/refault, media, signed-app repetition, full suites,
  packaging, notarization, and publication remain open.
- Overall v1.6.17 remains `PARTIAL / NOT RELEASE-READY`.

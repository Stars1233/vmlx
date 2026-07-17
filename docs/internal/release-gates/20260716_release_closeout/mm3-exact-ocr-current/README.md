# MiniMax-M3 exact OCR and media-cache boundary

Status: `PASS-LIVE` for scoped unambiguous OCR on Electron and Responses and
for the current image/video media-cache boundary; `PARTIAL` for ambiguous glyph
OCR, alternate-video isolation, and REAP variants.

## Runtime and UI ownership

- The visible Electron model selector chose `jangq-ai/MiniMax-M3-Coder-Small`.
  Starting it in single-model mode unloaded Bonsai PID 37342 and started only
  M3 PID 37842 on port 8017.
- The live argv uses the `minimax_m3` tool and reasoning parsers, paged cache,
  1,000 64-token blocks, and Block Disk L2. Health identifies the native cache
  as `minimax_m3_msa_v1` with three dense KV layers plus 57 sparse MSA layers;
  generic TurboQuant is correctly forced off because MSA index keys are part
  of the cache tuple.

## Exact OCR controls

- The first deterministic fixture deliberately kept digits and ASCII hyphens.
  M3 streamed ten content paints but returned `B81–MEDIA–B–DONE`, misreading
  `B1` and normalizing hyphens. The row is retained as a quality failure.
- The visually inspected unambiguous fixture contains only
  `MAGNOLIA CACHE DONE` in large bold sans-serif text. Electron returned that
  byte-exact marker twice, each time through seven progressive content paints,
  `finish_reason=stop`, no reasoning, and no tools. The screenshots and event
  records are `mm3-exact-ocr2.*` and `mm3-exact-ocr3-warm.*`.
- `mm3-exact-ocr-api.json` was produced by a real `curl -N` Responses request.
  It emitted seven timed `response.output_text.delta` events, matching done
  text, one `response.completed`, no warnings, and no reasoning events.

## Media-prefix boundary

The old unconditional media-cache bypass described by the first version of
this document is superseded by commit `8df1bfe86`. The current path forwards
raw image/video sources into the shared content-derived media key, salts
BlockAware fetch/store, rejects unsalted alternatives, and reuses an M3 prefix
only after it covers every media-token position. Shorter hits release and run
the original full vision prefill with pixels/grids still present.

Current Electron proof includes image A cold, 746-token RAM hit, same-shape B
zero-hit isolation, return-A RAM hit, and 746-token restart/L2 hit. A real MP4
then produced exact `BANANA8426` cold, on a 1,690-token RAM hit, and on a
1,690-token restart/L2 hit. Raw Responses emitted four progressive content
deltas, matching done text, and one completed terminal. The pre-fix
same-shape collision is retained rather than hidden. Exact source, rows,
screenshots, health, and open limits are in `../mm3-media-cache-current/`.

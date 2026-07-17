# MiniMax-M3 exact OCR and media-cache boundary

Status: `PASS-LIVE` for scoped unambiguous OCR on Electron and Responses;
`PARTIAL` for ambiguous glyph OCR, media-prefix reuse, and REAP variants.

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

The two identical unambiguous image requests each reported 742 prompt tokens
and zero cached tokens. That is consistent with the current source contract,
not evidence of a hidden hit:

- `BatchedEngine` preprocesses M3 image/video media into token IDs, pixel/video
  tensors, and grids before it enters the text engine (`engine/batched.py:2292-2308`).
- The scheduler recognizes that active payload on `SingleBatchGenerator`,
  resets `tokens_to_process` to the entire prompt, and sets
  `cache_to_use=None` (`scheduler.py:6112-6129`). The comment states why: every
  image-token position must be present in one forward for the vision splice.
- Health after the OCR rows reports zero scheduler cache hits/misses and zero
  disk blocks for these media requests. M3 text-cache proof therefore does not
  prove image/video cache reuse.

This is safe fail-closed behavior. Enabling M3 media reuse needs a family-owned
cached-vision boundary that preserves pixel/video content, grids, the MSA KV
tuple, media-key isolation, and restart reconstruction; a family-name allowlist
would be insufficient.

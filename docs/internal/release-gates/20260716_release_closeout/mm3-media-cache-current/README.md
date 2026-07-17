# MiniMax-M3 media-keyed prefix cache proof

Status: `PASS-LIVE` for this MiniMax-M3-Small image/video RAM and restart-L2
boundary; `PARTIAL` for ambiguous OCR, distinct-content video isolation, REAP,
and the broader M3 catalog.

## Source boundary

Commit `8df1bfe86` replaces the old unconditional M3-media cache bypass with a
content-keyed, family-owned boundary:

- `engine/batched.py` forwards the original image/video sources to
  `EngineCore`; the cache key is therefore derived from media content rather
  than only preprocessed tensor shape.
- `engine_core.py` uses the shared multimodal media-key helper and stamps the
  request with the resulting side key.
- `scheduler.py` includes that side key in BlockAware fetch/store. It rejects
  unsalted memory, legacy, and prompt-L2 alternatives for media requests.
- A cached M3 prefix is reusable only when it extends strictly beyond every
  image/video token. A shorter hit is released and the original full vision
  prefill is retried atomically with pixels and grids still present.
- The replay path omits vision tensors only after the cached prefix has crossed
  that boundary, seeds `all_tokens` from the cached prefix, and processes the
  remaining text tail.

Exact numbered excerpts are in `source-trace.txt`.

## Retained pre-fix failure

The first implementation salted only from preprocessed tensor metadata. Row
243 then restored 746/751 `paged` tokens from image A for a different,
same-shape image B and leaked A's exact `MAGNOLIA CACHE DONE` answer. That is a
real cross-media cache collision, retained in `m3-media-cache-b1-cold.png`.
The raw-source handoff above is the owning fix; no answer rewrite or model-name
exception was added.

## Deterministic Electron image controls

All rows below used the visible Electron app with Thinking Off and tools off:

| DB row | Payload | Result | Cache | TTFT |
|---:|---|---|---|---:|
| 246 | image A | exact `MAGNOLIA CACHE DONE` | cold, 0 cached | 4.97s |
| 249 | image A | exact | 746/751 `paged` | 1.10s |
| 252 | same-shape image B | `B1–MEDIA–B–DONE`; no A leak | cold, 0 cached | 2.64s |
| 255 | image A again | exact | 746/751 `paged` | 1.12s |
| 258 | image A after visible PID 63774 to 64620 restart | exact | 746/751 `paged+disk` | 3.45s |

Row 252 remains a strict OCR miss because ASCII hyphens became en dashes; its
zero-hit isolation and absence of the A marker are the cache claim.

## Deterministic Electron video controls

The fixture is the real six-frame MP4
`../gemma4-media-cache-current/gemma4-media-a-video.mp4`, visibly containing
`BANANA8426`. After clearing Prefix and L2, the deterministic controls used
temperature 0, Thinking Off, and tools off:

| DB row | Result | Cache | TTFT |
|---:|---|---|---:|
| 267 | exact `BANANA8426` | cold, 0 cached | 4.44s |
| 270 | exact | 1,690/1,695 `paged` | 1.32s |
| 273 | exact after visible PID 64620 to 65700 restart | 1,690/1,695 `paged+disk` | 3.50s |

Rows 261/264 are retained stochastic controls: the cold row read
`BANANA84226`; the RAM hit read `Fresh Image\nBANANA8426`. They show why the
deterministic settings were established rather than silently counting a
variable-temperature result as an exact proof.

## Raw Responses stream

A real `curl -N /v1/responses` request reused the same MP4/prompt with
temperature 0 and thinking disabled. It restored 1,690/1,695 `paged` tokens
and emitted progressive content deltas:

- 1.1615s `BAN`
- 1.2074s `ANA`
- 1.2486s `842`
- 1.2900s `6`
- 3.6707s `response.output_text.done` with exact `BANANA8426`
- 3.6710s one `response.completed`, output tokens 5

There were no reasoning events because thinking was deliberately off. The
visible content itself was progressive; the roughly 2.38s delay from the last
content delta to the terminal event is measured and retained. See
`api-video-responses.json`.

## Cache telemetry and validation

Final health records 1,690 indexed tokens, 27 disk blocks, 108 block-disk hits,
and two scheduler requests saving 3,380 tokens. Native cache telemetry remains
`minimax_m3_msa_v1`: three dense-KV layers and 57 sparse MSA/index layers;
generic TurboQuant remains off because the MSA index-key tuple is not generic
KV. `final-health.json` contains the complete response.

Current focused rerun: 207/207 tests passed with two third-party librosa
deprecation warnings; command and output summary are in `focused-tests.txt`.

## Open gates

- Prove a different-content video B miss and return-A hit; image A/B/A already
  proves the shared content-key implementation, but video B is not live-proven.
- Retain ambiguous digit/dash OCR as a model-quality `PARTIAL`; do not rewrite
  generated text.
- Exercise REAP only if it can be done without host-reboot risk.
- Run the full Python/panel, bundled-Python, clean-build, signing, notarization,
  and install-smoke gates before any release claim.

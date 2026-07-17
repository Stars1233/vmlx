# Bonsai 1-bit image-keyed hybrid cache proof

Status: `PASS-LIVE` for the scoped Bonsai image path at code head
`f993e36b8`; `PARTIAL` for Bonsai video and the wider Qwen/Bonsai catalog.

## Source ownership

- The real artifact is `Qwen3_5ForConditionalGeneration` with
  `model_type=qwen3_5`, `Qwen3VLProcessor`, a vision config, and distinct image
  and video token IDs.
- `MLLMBatchGenerator._media_prefix_cache_allowed` defaults the clean
  media-conditioned prefix path on only for config-derived Qwen 3.5 families,
  requires a stable media side-key, retains an explicit off switch, and leaves
  other families fail-closed (`vmlx_engine/mllm_batch_generator.py:4856-4924`).
- The side-key is installed before cache lookup, so the same text and token
  shape cannot alias different pixels (`mllm_batch_generator.py:5557-5615`).
- Cold media requests build the clean N-1 cache while the real pixel tensors
  are still live (`mllm_batch_generator.py:6387-6414`). The same keyed boundary
  owns the attention KV blocks and all native SSM/GDN companion layers; media
  prompts do not fall through to text-only async rederive
  (`mllm_batch_generator.py:6575-6673`).
- Bonsai Auto uses native TQ8 storage only for its 16 attention KV layers. Its
  48 path-dependent SSM/GDN layers remain native companion state and persist in
  the SSM disk store.

## Automated validation

The current-tree focused run is `focused-tests.txt`: 219 passed, six skipped,
and two third-party deprecation warnings across `test_zaya_runtime.py`,
`test_mllm_scheduler_cache.py`, and `test_ssm_companion_cache.py`. The contracts
include Qwen default-on/explicit-off policy, media tensor retention at the clean
prefill boundary, Qwen position-state restoration, media side-key storage, and
the non-Qwen fail-closed gate.

## Electron image A/B/return-A proof

Every clean comparison created a fresh visible Electron chat and explicitly
selected Thinking Off inside that chat. Built-in tools were off.

- Cold A: exact `Q27-EXACTONCE-ELECTRON2-DONE`; 14 progressive content paints;
  4,964 prompt tokens; no cached tokens; 21.33s TTFT.
- Identical A: exact marker; 14 progressive paints; 4,963 tokens restored as
  `paged+ssm`; 0.69s TTFT.
- Same-shape deterministic B: zero cached tokens and exact
  `B1-MEDIA-B-DONE` in seven progressive paints. No A marker leaked.
- Return A: exact A marker; 4,963 `paged+ssm` tokens; 14 progressive paints;
  0.66s TTFT.
- Visible Electron Stop/Start changed PID 34884 to 36409 without clearing L2.
  A then restored all 4,963 reusable tokens as `paged+ssm+disk`, returned the
  exact marker in 14 progressive paints, and reached TTFT in 1.64s.

After the disk row, health reported 78 block-disk/native-TQ hits, zero selected
prefix block misses, and one SSM-companion disk hit. The visible screenshot
`b1-media-a6-disk.png` shows the real attached screenshot, exact answer,
Thinking Off, and `4963 paged+ssm+disk cached` in the Electron UI.

## Raw Responses API proof

`api-warm.json` was produced by a real `curl -N` request to `/v1/responses`
using the same prompt and image as Electron. It reused the Electron-created
prefix, returned exact `Q27-EXACTONCE-ELECTRON2-DONE`, emitted 14 timed
`response.output_text.delta` events followed by one `response.completed`, and
reported 4,963 cached `paged+ssm` input tokens. There were no reasoning deltas
because the request explicitly set `enable_thinking=false`.

## Retained failures and remaining gates

- The first Auto-thinking image row is retained as a quality failure: it
  streamed 1,024 reasoning paints, then stopped after 31 visible content paints
  with the marker truncated. This is not counted as a media/cache pass.
- A second screenshot with small UI text was a cache miss but OCR returned only
  `DONE`; the deterministic large-marker B row replaced it for the isolation
  proof rather than hiding the miss.
- Bonsai video needs its own A/B/return/restart proof. Qwen 3.6 video evidence
  does not clear Bonsai or other media families.

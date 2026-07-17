# Qwen 3.6 27B media-keyed hybrid cache proof

Status: `PASS-LIVE` for the scoped Qwen 3.6 image path at code commit
`9982d9ae2`. This does not clear video or other VLM families.

## Source ownership

- `MLLMBatchGenerator._media_prefix_cache_allowed` enables the clean
  media-conditioned path by default only for config-derived Qwen 3.5/3.6
  families, keeps an explicit off switch, and leaves other families behind
  their existing double opt-in.
- The generator builds the N-1 cache while pixel/video tensors and grids are
  still live. The previous deferred location ran after those tensors were
  cleared and could only have produced a false text-only cache.
- The clean cache supplies both attention KV blocks and the 48 native
  SSM/GDN companion layers under the same SHA-256 media side-key. The
  auxiliary prefill restores Qwen's active `_rope_deltas` and
  `_position_ids`.
- The scheduler stores attention layers with
  `store_cumulative_state=False`; block-disk L2 uses its native TQ codec,
  while the SSM companion store owns the full-precision path-dependent state.

Exact numbered source excerpts are in `source-trace.txt`.

## Automated proof

`focused-tests.txt` records 218 passed, 6 skipped, and two third-party
deprecation warnings across:

- `tests/test_zaya_runtime.py`
- `tests/test_mllm_scheduler_cache.py`
- `tests/test_ssm_companion_cache.py`

The new contracts cover Qwen default-on/explicit-off policy, live pixel
retention at the clean prefill boundary, Qwen position-state restoration,
media side-key storage, and the non-Qwen fail-closed gate.

## Raw API live proof

All A/B rows used the same prompt and 2800x1800 screenshots, producing the
same 4,956-token prompt shape.

- Cold A: first content 26.6666s (includes the one-time clean media N-1
  prefill), 36 progressive content deltas, exact Qwen marker/model, stop.
- Warm A: restored 4,955 tokens as `paged+ssm`, first content 1.0724s,
  35 progressive deltas, exact output.
- Different-image B: no cached-token hit, first content 26.5665s, and read
  only the MiniMax marker/model. No A output leaked.
- Return A: restored 4,955 `paged+ssm` tokens at 1.1049s and returned A.
- Bypass A: `skip_prefix_cache=true` forced an uncached 17.2628s first
  content without replacing the saved A entry.
- Post-bypass A: restored 4,955 tokens at 1.0722s.
- Process restart: Electron stopped PID 4659 and started PID 5791 without
  clearing L2. Before the request, RAM held zero cached tokens while block
  disk held 469 blocks and SSM disk held 68 entries. A then restored 4,955
  tokens as `paged+ssm+disk` at 1.0741s, streamed 35 deltas, and returned
  the exact image-derived answer.
- Post-hit health records 78 block-disk/native-TQ hits, one SSM companion
  disk hit, zero disk misses for the selected prefix, and a successful
  0.398954s worker reconstruction.

## Electron live proof

A fresh visible chat inherited built-in tools off and attached the real A PNG.
The renderer emitted 68 reasoning updates and 33 content updates, no tool
events, and one `stop` completion. The final bubble visibly shows the exact
marker/model and `4955 paged+ssm cached`, 1.06s TTFT.

`electron-db-row.json` independently records the full image data URL in the
user row and an assistant row with 407 reasoning characters, exact final
content, no tools/warnings, and the same cache metrics.

## Remaining gates

- Qwen video cache needs an equivalent cold/warm/cross-video/restart proof.
- MiniMax M3, Gemma 4, Bonsai, Step, and other advertised media families need
  family-owned cache proofs; this change deliberately does not enable them.
- Full-suite, packaging, signing, notarization, and release gates remain open.

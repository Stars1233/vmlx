# Prompt-disk immediate-Stop and first-turn role durability

Date: 2026-07-19
Source commit: `7a146eefbfd5a4f8f8ac692a22d890358a5a77aa`
Model: `jangq-ai/openPangu-2.0-Flash-JANG_3M`
Scoped status: **FIXED_SOURCE + VERIFIED-LIVE**
Overall release status: **PARTIAL / NOT A 1.6.12 RELEASE CANDIDATE**

This gate covers two shared prompt-disk defects exposed by an Electron
Stop/Start immediately after a terminal answer while the 10 GB L2 directory
was already near its size ceiling. It is not an openPangu-specific parser,
quant, or sampling repair.

## Defect 1: shutdown could cancel terminal cache cleanup

`EngineCore` and `MLLMScheduler` intentionally dispatch terminal output before
worker-owned prefix/TQ/SSM/media cleanup so the UI can paint the final delta
without waiting for persistence. Both stop methods then cancelled their
processing task immediately. If Electron Stop arrived in that interval, the
visible answer existed but the cache snapshot could be lost before it was
queued to `DiskCacheManager`.

The existing `_terminal_cleanup_complete` event already gated next-request
admission. Commit `7a146eefb` makes it the shutdown durability boundary too:
text and MLLM stop paths wait up to five seconds for an in-flight terminal
cleanup before cancelling their loop, then execute the existing disk-cache
shutdown/flush. Active, non-terminal generation remains cancellable.

Focused concurrency tests force the losing race and prove both stop paths wait
for cleanup before cancellation. They also preserve the existing
terminal-before-cleanup streaming order.

## Defect 2: a one-message user base was tagged assistant

`BatchedEngine._compute_segment_boundaries()` returned no boundary for any
one-message conversation. The full prompt therefore fell back to
`cache_type=assistant`. Prompt-disk eviction deliberately prioritizes
assistant entries ahead of user/system entries. At the full 10 GB ceiling, a
new first-turn base could consequently write successfully and then evict
itself before the first follow-up, while older user entries survived.

The pre-fix control is retained in
`prefix-pre-role-fix-shutdown-logs.txt`: the 1,539-token snapshot logged a
successful store followed by one eviction. The next process loaded the six
old entries (6,810 tokens), did not contain the 1,539-token record, and the
1,612-token follow-up missed L2. Its final answer was coherent and exact, so
this was cache classification/eviction, not model output corruption.

Commit `7a146eefb` now computes a real boundary for a single user or system
message. Empty and assistant-only histories retain the safe fallback. New
behavior tests pin all three cases.

## Validation

`focused-tests.txt` records 119/119 current-source tests passing across:

- terminal dispatch/cleanup and shutdown ordering;
- first-turn role boundaries;
- disk shutdown, N-1 payload-prefix matching, and eviction support;
- MiniMax-M3 typed cache paths;
- native TurboQuant disk encoding/decoding;
- openPangu typed composite cache round trips.

No prompt rewrite, hidden output cap, sampler clamp, fabricated tool argument,
or model-family exception was introduced.

## Live Electron proof at a full L2 ceiling

The real Electron dev app on CDP 9335 loaded openPangu through the Sessions
Start control. The visible header showed the JANG affine `JANG_3M` artifact,
port 8027, and the live PID. This is not JANGTQ/MXTQ or base MLX MXFP. The
launch retained its architecture-owned composite cache, `--no-paged-cache`,
prompt disk enabled, generic TQ disabled, reasoning Auto, and tools disabled
for the cache-specific chat.

A fresh one-message base produced 824 separate reasoning characters and exact
visible `OP-ROLE-EVICT-BASE-DONE`. The harness detected the persisted terminal
row in a 200 ms poll and immediately clicked the visible Stop control. After
shutdown, the SQLite index contained the new 1,322-token, 1,786.8 MB entry as
`cache_type=user`; the older 1,582-token LRU entry had been evicted instead.
`index-after-immediate-stop.txt` is the direct post-stop index proof.

After visible Start replaced the process, the same Electron chat asked for
fact 57. It:

- restored 1,321/1,395 prompt tokens from `disk`;
- had zero resident L1 prefix bytes before the restore;
- produced a fresh 952-character reasoning rail;
- progressively painted visible content in the retained 2 s/6 s screenshots;
- exact-finaled `OP-ROLE-EVICT-RESTORE-DONE STATE-3547` with no warning;
- reported 0.70 s TTFT and a live disk hit.

`electron-rows.json`, the screenshots, `restore-logs.txt`, and
`electron-restore-health.json` preserve the UI/DB/log/health evidence.

## Raw Responses after an independent UI restart

A detached on-box curl was used so SSH command lifetime could not truncate the
client. It restored 1,321 input tokens from disk, emitted 463 separate
reasoning-summary deltas and 19 progressive content deltas, exact-finaled
`OP-ROLE-RESPONSES-DETACHED-DONE STATE-10689`, and emitted
`response.completed` with status `completed`.

An earlier SSH-coupled capture ended mid-reasoning even though the server later
finished and stored the request. It is retained as
`invalid-ssh-transport-truncated-responses.sse` and is explicitly not counted
as server failure or pass evidence. The detached client falsified server-side
truncation for this row.

The request opted into the existing vMLX `response.usage` stream extension and
received 483 such events in addition to terminal usage. Whether that extension
matches the current public Responses protocol remains a separate protocol-
parity audit; this gate does not silently classify it as standard.

## Raw Chat after another independent UI restart

Detached Chat Completions restored the same 1,321-token disk prefix and emitted
512 non-empty `reasoning_content` chunks plus 17 progressive content chunks.
The final was exactly `OP-ROLE-CHAT-DETACHED-DONE STATE-14260`. Wire order was:

1. ordinary chunks with `usage:null`;
2. `finish_reason=stop` at JSON index 530;
3. exactly one choices-empty total-usage chunk at index 531;
4. exactly one `[DONE]`.

No ordinary content/reasoning chunk carried non-null usage.

## Remaining boundary

This closes immediate-stop durability and one-message priority eviction for
the shared prompt-disk path. openPangu intentionally cannot prove generic paged
block reuse, block-disk refault, or generic q4/q8 TQ because its native
path-dependent MLA/DSA/SWA/causal-conv composite cache disables those paths.
Those rows remain assigned to compatible models. The full model/protocol/media/
gateway matrix, full suites/build, bundled-Python refresh, signing,
notarization, and release publication remain separate gates.

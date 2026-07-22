# Global non-Laguna reasoning source audit

Date: 2026-07-21 (America/Los_Angeles)

This was a read-only family audit. It found source risks and did not run the
affected models. Laguna was excluded and untouched. The audit was performed at
`e16a3dac1`; the relevant files were unchanged through `ad931a45f`.

Verdict: `PARTIAL / RELEASE BLOCKED` until the high-priority rows below are
fixed and live-proven.

## High-priority findings

1. **Ollama streaming bypasses family-specific thinking normalization.**
   `vmlx_engine/server.py`'s streaming Ollama path resolves a boolean but does
   not apply MiniMax-M3 `thinking_mode` normalization or Mistral4's
   boolean-to-`reasoning_effort` mapping. Non-streaming delegates through Chat
   and does receive that normalization. Required proof: MiniMax-M3 and
   Mistral4 Ollama stream/non-stream Auto/On/Off plus a tool turn.

2. **Anthropic streaming does not safely handle visible text followed by late
   reasoning.** `vmlx_engine/api/anthropic_adapter.py` can open a thinking
   block while a text block is still open and reuse the content index. Gemma4
   explicitly supports late thought after visible content. Required proof:
   Gemma4 visible-text -> late-reasoning -> final/tool with balanced,
   monotonically indexed Anthropic content-block events.

3. **Ollama streaming thinking-Off retains historical private reasoning.**
   `vmlx_engine/api/ollama_adapter.py` maps historical `message.thinking` to
   `reasoning_content`, while the streaming server path forwards the messages
   without the Off-mode history stripping used by Chat, Responses, and
   Anthropic. Required proof: two-turn Qwen/Bonsai Ollama stream, then
   `think:false`, with no replayed reasoning or raw tags.

## Medium-priority findings

- Generic ThinkParser and MiniMax-M3 streaming parsers do not hold partial
  delimiter prefixes across tokenizer chunk boundaries. Split `<think>`,
  `[THINK]`, or `<mm:think>` markers can leak fragments. Gemma4 already has a
  partial-marker holdback implementation to use as a reference.
- Tool markup parsed from a private reasoning rail can promote cleaned
  reasoning residue to visible output in edge paths. A reasoning-only tool
  emission row is needed for Chat and Responses.

## Source-backed paths, not live closure

- Chat and Responses have separate reasoning and visible-content event paths.
- Panel main-process consumers and renderer maintain separate reasoning,
  content, and tool surfaces and flush SSE lines incrementally.
- Family registry mappings exist for Qwen, Mistral, DSV4, Step, Gemma4,
  Nemotron, LFM, MiniMax M2, and MiniMax M3.
- Bonsai has no standalone registry family; its behavior must be grounded in
  the live bundle's real `model_type` and config before inheriting a Qwen
  claim.

These source traces must not be described as verified runtime behavior.

## Smallest release-closing matrix

1. MiniMax-M3 Ollama stream/non-stream Auto/On/Off plus required tool.
2. Mistral4 Ollama stream/non-stream On/Off and
   `reasoning_effort=high/none`.
3. Gemma4 Anthropic visible -> late thought -> final/tool event balance.
4. Qwen3/Bonsai two-turn Ollama Off history-stripping plus required tool.
5. DSV4 Responses Auto/On/Off plus required DSML tool.
6. LFM or Step Auto, intentional Off rejection, and tool continuation.
7. MiniMax-M2 streaming On/Off plus tool.

The LFM Chat/Responses subset is separately live-proven in
`../20260722_lfm_reasoning_tool_stream/`; it does not close the remaining
family rows above.

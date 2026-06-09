# Responses Long Tool Cache Gate

- model: `qwen3.6-35b-a3b-mxfp8-mtp-long-tool`
- target_chars_per_turn: `6000`
- turns: `3`

- final_turn_no_tools: `True`

- final_turn_disable_thinking: `False`

- resolve_tool_calls_in_turn: `True`

- tool_choice_mode: `required`

- resolution_tool_choice: `none`

- require_tool_call_each_turn: `True`

- require_tool_evidence: `True`

- require_cache_each_turn_after_first: `True`

## Acceptance

- overall_pass: `True`
- turns_completed: `True`
- previous_response_id_used: `True`
- cache_reuse_observed: `True`
- require_cache_each_turn_after_first: `True`
- cache_reuse_each_turn_after_first: `True`
- tool_call_observed: `True`
- require_tool_call_each_turn: `True`
- tool_call_each_required_turn: `True`
- require_tool_evidence: `True`
- tool_evidence_each_required_turn: `True`
- final_turn_tools_disabled: `True`
- final_turn_thinking_disabled: `True`
- visible_or_tool_output_each_turn: `True`
- visible_output_observed: `True`
- final_turn_visible_output: `True`
- no_loop_like_tail: `True`
- no_tool_markup_leak: `True`

## Rows

| turn | elapsed_s | cached_tokens | visible_chars | reasoning_chars | function_calls | warnings | loop_like | tool_markup_leak | tool_grounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 6.483 | 0 | 1059 | 0 | 1 | 0 | False | False | True |
| 2 | 7.831 | 128 | 1162 | 0 | 1 | 0 | False | False | True |
| 3 | 8.935 | 256 | 2002 | 0 | 0 | 0 | False | False | False |

Raw response, cache, health, and tail_review files are preserved next to this summary.

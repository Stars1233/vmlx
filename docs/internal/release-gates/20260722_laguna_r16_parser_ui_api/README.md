# 2026-07-22 Laguna parser/reasoning UI/API proof

Status: `PARTIAL / RELEASE-CRITICAL`.

Source checkpoint: `52b8d05786afd158e7c67fadee4a15d394a73790`.

Live app: Electron dev app on `erics-m5-max.local`, CDP `9335`, gateway
`127.0.0.1:8088`.

Model:
`/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_4M`.

## Parser provenance

Bundle facts:

- `config.json`: `model_type=laguna`, architecture `LagunaForCausalLM`
- `generation_config.json`: `reasoning_parser=poolside_v1`,
  `default_chat_template_kwargs.enable_thinking=true`
- `jang_config.json`: `chat.reasoning.parser=deepseek_r1`,
  `chat.tool_calling.parser=glm47`,
  `chat.vendor_parsers.{reasoning,tool}=poolside_v1`
- `chat_template.jinja` contains `enable_thinking`, `reasoning_content`,
  `<think>`, and `</think>` branches.

Runtime argv for PID `56839` used `--tool-call-parser glm47` and
`--reasoning-parser deepseek_r1`. Current engine source registers
`poolside_v1` as aliases to these effective parser classes. Therefore the
argv spelling is not itself proof of a defect; live parser behavior is the
acceptance criterion.

## Electron UI proof

Fresh visible Laguna chat, not old history.

Visible settings matched bundle defaults:

- Enable Thinking: `Auto`
- Temperature: `1.00`
- Top P: `1.00`
- Top K: `20`
- Tools: enabled
- Wire format: `Responses`

Rows:

- Reasoning/answer: separate Reasoning rail `2305 chars`; visible answer
  `The larger product is 56 times 14.`; no native marker leak; `718 tokens`,
  `49.9 t/s`, `0.60s TTFT`, `15.0s total`.
- Tool: separate Reasoning rail `79 chars`; exactly one visible
  `Info panel/package.json` tool card; exact visible final
  `The package file is 5.2 KB.`; prompt showed
  `4861 block-disk+tq-native cached`; no marker leak.

UI verdict: `PASS_SCOPED` for reasoning separation, answer emission, tool
parser, and disk-only cache reuse in this fresh chat.

## Gateway API proof

Artifacts:

- `laguna-api-gateway-proof.json`
- `laguna-api-terminal-addendum.json`
- `laguna-anthropic-ollama-gateway-proof.json`

Rows:

| Case | Result |
|---|---|
| Chat Auto/default hard prompt | `reasoning_content` `2118 chars`, no native marker leak, but visible answer over-generated explanation and hit `length` |
| Responses explicit On hard prompt | reasoning `2214 chars`, no native marker leak, but visible answer over-generated explanation and ended `response.incomplete` |
| Chat required tool | emitted tool-call chunks and `tool_calls` terminal; no visible marker leak; no reasoning on this short tool prompt |
| Chat tool-result continuation | exact visible `The package file is 5.2 KB.`, `stop + DONE`; no reasoning on this short tool prompt |
| Chat Auto/default terminal addendum | `reasoning_content` `1665 chars`, visible `30 times 25 is larger.`, `stop + DONE`, no marker leak |
| Responses explicit On terminal addendum | visible `43 × 18 is larger.`, `response.completed`, no marker leak, but `0` reasoning chars |
| Anthropic hard prompt | protocol-native thinking deltas `2799 chars`, text deltas `150 chars`, `message_stop`, no marker leak; visible answer over-generated instead of exact |
| Anthropic required tool | exact `file_info({"path":"panel/package.json"})`, `message_stop`, no visible content before tool |
| Anthropic tool continuation | exact visible `The package file is 5.2 KB.`, progressive text deltas, `message_stop`, no marker leak |
| Ollama hard prompt | `thinking` deltas `2581 chars`, content deltas `1123 chars`, terminal `stop`, no marker leak; visible answer over-generated instead of exact |
| Ollama required tool | exact `file_info({"path":"panel/package.json"})`, terminal `tool_calls`, no visible content before tool |
| Ollama tool continuation | exact visible `The package file is 5.2 KB.`, progressive content deltas, terminal `stop`, no marker leak |

API verdict: parser separation and tool parser are live, but the Laguna
strict-format reasoning rows are not fully closed. Current evidence shows
model-owned variable/no-reasoning on short prompts plus hard-prompt reasoning
rows that sometimes overrun the requested concise visible answer. Do not claim
the full Laguna strict-format API reasoning gate green from this proof alone.

## Remaining Laguna work

- Bounded Responses explicit-On row that both emits reasoning and reaches
  `response.completed` with a concise visible answer.
- Bounded Anthropic/Ollama strict-format hard rows that both reason and obey
  the requested concise answer. Transport/parser/tool loop is live, but strict
  visible formatting is still partial.
- Longer agentic tool loop and cancellation/recovery.
- Low-limit cache eviction/refault and corrupt/missing companion fallback.
- Signed-app repeat after packaging.

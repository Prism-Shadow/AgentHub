# DeepSeek V4: no chat template, an encoding module instead

> Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/encoding_dsv4.py,
> https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/encoding_dsv4.py and
> https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/main/encoding/encoding_dsv4.py
> (snapshot 2026-09-04)

The three DeepSeek V4 models served through `openai_chat_vllm_adapter` publish **no chat
template**. `chat_template.jinja` returns HTTP 404 in all three repositories, and
`tokenizer_config.json` (801 bytes) carries no `chat_template` key. Nothing in those
repositories is a template, so there is nothing to snapshot into `../chat_templates/`.

The prompt format is instead defined by a reference implementation, `encoding/encoding_dsv4.py`,
documented by `encoding/README.md` beside it. It is executable Python rather than an inert
template, and most of it covers tokenization, tool rendering and completion parsing that has no
bearing on request parameters, so it is cited and quoted here rather than vendored.

| Model | Upstream file | Size | License | MD5 |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Pro` | [encoding/encoding_dsv4.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/encoding_dsv4.py) | 27908 B | [MIT](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/LICENSE) | `f5e65effdf98...` |
| `deepseek-ai/DeepSeek-V4-Flash` | [encoding/encoding_dsv4.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/encoding_dsv4.py) | 27908 B | [MIT](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/LICENSE) | `f5e65effdf98...` |
| `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | [encoding/encoding_dsv4.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/main/encoding/encoding_dsv4.py) | 36707 B | [MIT](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/main/LICENSE) | `06a743aa6802...` |

`DeepSeek-V4-Pro` and `DeepSeek-V4-Flash` ship the same file, byte for byte.
`DeepSeek-V4-Flash-Vision-Exp` does not, and the difference is not confined to the image
handling its name implies — **it changes the accepted `reasoning_effort` values**.

## The parameters

Both variants expose the same two knobs, on `encode_messages()` and the `render_message()` it
calls per message:

- `thinking_mode`, a string that must be `"chat"` or `"thinking"` — this is the on/off switch,
  and it has no default: `assert thinking_mode in ["chat", "thinking"]`.
- `reasoning_effort`, an optional string whose accepted values differ per variant.

There is no boolean `thinking` parameter and no `enable_thinking` anywhere in the module; the
`chat_template_kwargs` the adapter sends are the vLLM-side spelling of these two knobs, not
names this module reads directly.

### Pro and Flash accept only `high`, `max` and `None`

```python
    # Reasoning effort prefix (only at index 0 in thinking mode with max effort)
    assert reasoning_effort in ['max', None, 'high'], f"Invalid reasoning effort: {reasoning_effort}"
    if index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max':
        prompt += REASONING_EFFORT_MAX
```

`"low"` is **not** in the accepted set and trips the assertion. And of the values that do pass,
only `'max'` changes the prompt: `'high'` and `None` both fall through to no prefix, so they are
indistinguishable in the rendered output.

### Flash-Vision-Exp accepts `low`, `high` and `max`

```python
# Reasoning effort levels. In thinking mode, the prompt for the selected level is
# prepended at the very beginning of the conversation. `low` is the default and
# adds nothing.
REASONING_EFFORT_PROMPTS: Dict[str, str] = {
    "low": "",
    "high": (...),
    "max": (...),
}
DEFAULT_REASONING_EFFORT = "low"
```

```python
    # Reasoning effort prefix (only at index 0 in thinking mode; "low" adds nothing)
    reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    assert reasoning_effort in REASONING_EFFORT_PROMPTS, \
        f"Invalid reasoning effort: {reasoning_effort}, expected one of {list(REASONING_EFFORT_PROMPTS)}"
    if index == 0 and thinking_mode == "thinking":
        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]
```

Here `low` is the documented default and contributes an empty prefix, `high` carries the text
that Pro and Flash call `REASONING_EFFORT_MAX`, and `max` adds a stronger prefix that has no
counterpart in the other two files. Effort levels therefore do not line up across the three
models even where the names match: `high` on Flash-Vision-Exp is roughly `max` on Pro.

In both variants the prefix is emitted only at `index == 0` and only in thinking mode, so
`reasoning_effort` is inert whenever `thinking_mode == "chat"`.

## Related

DeepSeek's hosted API documents `reasoning_effort` as `low`/`high`/`max` for all V4 models; see
[../../deepseek_v4/docs/thinking-mode.md](../../deepseek_v4/docs/thinking-mode.md). That is the
platform contract, not the open-weights one recorded above, and the two disagree for
`DeepSeek-V4-Pro` and `DeepSeek-V4-Flash`.

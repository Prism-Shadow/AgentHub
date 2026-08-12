# Model Selection

Use exact model IDs. If a model ID is not listed, ask the user to confirm the exact ID before using it. The supported model entries (base URL, client, modalities, context window, USD/CNY pricing) are also available at runtime via `listSupportedModels()`.

| Family | Provider | Model IDs | API Key | Base URL |
| --- | --- | --- | --- | --- |
| Gemini 3 | Official / Vertex AI | `gemini-3.1-pro-preview`, `gemini-3.5-flash`, `gemini-3.1-flash-lite` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini 3.6 | Official / Vertex AI | `gemini-3.6-flash`, `gemini-3.5-flash-lite` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini 3 Image | Official / Vertex AI | `gemini-3.1-flash-image`, `gemini-3-pro-image` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini 3 TTS | Official / Vertex AI | `gemini-3.1-flash-tts-preview` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini Embedding | Official / Vertex AI | `gemini-embedding-2` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Claude 4.6 | Official / ModelVerse | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.6 | Bedrock | `global.anthropic.claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.7 | Official / ModelVerse | `claude-opus-4-7` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.7 | Bedrock | `global.anthropic.claude-opus-4-7` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.8 | Official / ModelVerse | `claude-opus-4-8` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.8 | Bedrock | `global.anthropic.claude-opus-4-8` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 5 | Official / ModelVerse | `claude-fable-5` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 5 | Bedrock | `global.anthropic.claude-fable-5` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| GPT 5.4 | Official / ModelVerse | `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| GPT 5.5 | Official / ModelVerse | `gpt-5.5` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| OpenAI Embedding | Official | `text-embedding-3-small`, `text-embedding-3-large` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| Kimi-K2.6 | Official | `kimi-k2.6` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Kimi-K2.6 | OpenRouter | `moonshotai/kimi-k2.6` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Kimi-K2.6 | SiliconFlow | `Pro/moonshotai/Kimi-K2.6` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Kimi-K3 | Official | `kimi-k3` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Kimi-K3 | OpenRouter | `moonshotai/kimi-k3` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| DeepSeek V4 | Official | `deepseek-v4-pro`, `deepseek-v4-flash` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| DeepSeek V4 | OpenRouter | `deepseek/deepseek-v4-pro`, `deepseek/deepseek-v4-flash` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| DeepSeek V4 | SiliconFlow | `deepseek-ai/DeepSeek-V4-Pro`, `deepseek-ai/DeepSeek-V4-Flash` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| GLM-5.1 | Official | `glm-5.1` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5.1 | OpenRouter | `z-ai/glm-5.1` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5.1 | SiliconFlow | `Pro/zai-org/GLM-5.1` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5.2 | Official | `glm-5.2` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5.2 | OpenRouter | `z-ai/glm-5.2` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5.2 | SiliconFlow | `zai-org/GLM-5.2` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| MiniMax-M3 | Official | `MiniMax-M3` | `MINIMAX_API_KEY` | `MINIMAX_BASE_URL` |

Common gateway base URLs:

- OpenRouter: `https://openrouter.ai/api/v1`
- SiliconFlow: `https://api.siliconflow.cn/v1`
- ModelVerse: `https://api.modelverse.cn/v1` (`https://api.modelverse.cn/` for Claude)
- vLLM: `http://127.0.0.1:8000/v1/`

For models accessed through OpenAI-compatible APIs (e.g., Qwen series models via SiliconFlow or OpenRouter), pass `clientType: "openai"` (`clientType: "openai-embedding"` for embedding endpoints). These models use `OPENAI_API_KEY` and `OPENAI_BASE_URL`:

```typescript
const client = new AutoLLMClient({ model: "Qwen/Qwen3-Embedding-0.6B", clientType: "openai-embedding" });
```

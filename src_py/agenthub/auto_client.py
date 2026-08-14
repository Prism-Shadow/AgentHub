# Copyright 2025 Prism Shadow. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, AsyncIterator

from .abort_signal import AbortSignal
from .base_client import LLMClient
from .types import UniConfig, UniEvent, UniMessage


class AutoLLMClient(LLMClient):
    """
    Auto-routing LLM client that dispatches to appropriate model-specific client.

    This client is stateful - it knows the model name at initialization and maintains
    conversation history for that specific model.
    """

    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None, client_type: str | None = None
    ):
        """
        Initialize AutoLLMClient with a specific model.

        Args:
            model: Model identifier (determines which client to use)
            api_key: Optional API key
            base_url: Optional base URL for API requests
            client_type: Optional client type override
        """
        self._client = self._create_client_for_model(model, api_key, base_url, client_type)

    def _create_client_for_model(
        self, model: str, api_key: str | None = None, base_url: str | None = None, client_type: str | None = None
    ) -> LLMClient:
        """Create the appropriate client for the given model."""
        client_type = (client_type or os.getenv("CLIENT_TYPE") or model).lower()
        if client_type == "minimax-m3":
            from .minimax_m3 import MiniMaxM3Client

            return MiniMaxM3Client(model=model, api_key=api_key, base_url=base_url)
        # every Gemini generation shares the unified client ("gemini-3" also matches the
        # gemini-3.7/gemini-3.6/gemini-3.5-flash-lite client types)
        if any(
            prefix in client_type for prefix in ("gemini-3", "gemini-embedding")
        ):  # e.g., gemini-3.7-flash, gemini-3-flash-preview, gemini-embedding-2
            from .gemini3_7 import Gemini3_7Client

            return Gemini3_7Client(model=model, api_key=api_key, base_url=base_url)
        elif "claude" in client_type and (
            "4-7" in client_type or "4-8" in client_type or "-5" in client_type
        ):  # e.g., claude-opus-4-7
            from .claude5 import Claude5Client

            return Claude5Client(model=model, api_key=api_key, base_url=base_url)
        elif "claude" in client_type and "4-6" in client_type:  # e.g., claude-sonnet-4-6
            from .claude4_6 import Claude4_6Client

            return Claude4_6Client(model=model, api_key=api_key, base_url=base_url)
        elif "gpt-5.4" in client_type or "gpt-5.5" in client_type:  # e.g., gpt-5.5
            from .gpt5_5 import GPT5_5Client

            return GPT5_5Client(model=model, api_key=api_key, base_url=base_url)
        elif "glm-5" in client_type:  # the whole GLM series shares the unified client
            from .glm5_3 import GLM5_3Client

            return GLM5_3Client(model=model, api_key=api_key, base_url=base_url)
        elif "kimi-k3" in client_type:
            from .kimi_k3 import KimiK3Client

            return KimiK3Client(model=model, api_key=api_key, base_url=base_url)
        elif "kimi-k2.5" in client_type or "kimi-k2.6" in client_type:
            from .kimi_k2_6 import KimiK2_6Client

            return KimiK2_6Client(model=model, api_key=api_key, base_url=base_url)
        elif "deepseek-v4" in client_type:
            from .deepseek_v4 import DeepSeekV4Client

            return DeepSeekV4Client(model=model, api_key=api_key, base_url=base_url)
        elif "openai" in client_type and "embedding" in client_type:
            from .openai_embedding import OpenaiEmbeddingClient

            return OpenaiEmbeddingClient(model=model, api_key=api_key, base_url=base_url)
        elif "openai" in client_type and "embedding" not in client_type:
            from .openai import OpenaiClient

            return OpenaiClient(model=model, api_key=api_key, base_url=base_url)
        else:
            raise ValueError(
                f"{client_type} is not supported. "
                "Supported client types: minimax-m3, gemini-3.7, gemini-3.6, gemini-3, "
                "claude-5, claude-4-8, claude-4-7, claude-4-6, gpt-5.5, gpt-5.4, glm-5.3, glm-5.2, glm-5.1, kimi-k3, "
                "kimi-k2.6, kimi-k2.5, "
                "deepseek-v4, openai-embedding, openai."
            )

    def transform_uni_config_to_model_config(self, config: UniConfig) -> Any:
        """Delegate to underlying client's transform_uni_config_to_model_config."""
        return self._client.transform_uni_config_to_model_config(config)

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> Any:
        """Delegate to underlying client's transform_uni_message_to_model_input."""
        return self._client.transform_uni_message_to_model_input(messages)

    def transform_model_output_to_uni_event(self, model_output: Any) -> UniEvent:
        """Delegate to underlying client's transform_model_output_to_uni_event."""
        return self._client.transform_model_output_to_uni_event(model_output)

    async def _streaming_response_internal(
        self,
        messages: list[UniMessage],
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        raise NotImplementedError("Please use streaming_response instead.")

    async def streaming_response(
        self,
        messages: list[UniMessage],
        config: UniConfig,
        signal: AbortSignal | None = None,
    ) -> AsyncIterator[UniEvent]:
        """Route to underlying client's streaming_response."""
        async for event in self._client.streaming_response(
            messages=messages,
            config=config,
            signal=signal,
        ):
            yield event

    async def streaming_response_stateful(
        self,
        message: UniMessage,
        config: UniConfig,
        signal: AbortSignal | None = None,
    ) -> AsyncIterator[UniEvent]:
        """Route to underlying client's streaming_response_stateful."""
        async for event in self._client.streaming_response_stateful(
            message=message,
            config=config,
            signal=signal,
        ):
            yield event

    def clear_history(self) -> None:
        """Clear history in the underlying client."""
        self._client.clear_history()

    def get_history(self) -> list[UniMessage]:
        """Get history from the underlying client."""
        return self._client.get_history()

    def set_history(self, history: list[UniMessage]) -> None:
        """Set history in the underlying client."""
        self._client.set_history(history)

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

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List

from .types import UniConfig, UniEvent, UniMessage


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All model-specific clients must inherit from this class and implement
    the required abstract methods for complete SDK abstraction.
    """

    @abstractmethod
    def transform_uni_config_to_model_config(self, config: UniConfig) -> Any:
        """
        Transform universal configuration to model-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Model-specific configuration object
        """
        pass

    @abstractmethod
    def transform_uni_message_to_model_input(self, messages: List[UniMessage]) -> Any:
        """
        Transform universal message format to model-specific input format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            Model-specific input format (e.g., Gemini's Content list, OpenAI's messages array)
        """
        pass

    @abstractmethod
    def transform_model_output_to_uni_event(self, model_output: Any) -> UniEvent:
        """
        Transform model output to universal event format.

        Args:
            model_output: Model-specific output object (streaming chunk)

        Returns:
            Universal event dictionary
        """
        pass

    def transform_uni_event_to_uni_message(self, events: List[UniEvent]) -> UniMessage:
        """
        Transform a stream of universal events into a single universal message.

        This is a concrete method implemented in the base class that can be reused
        by all model clients. It accumulates events and builds a complete message.

        Args:
            events: List of universal events from streaming response

        Returns:
            Complete universal message dictionary
        """
        text_content = ""
        thought_signature = None
        content_items = []

        for event in events:
            if event["type"] == "text":
                text_content += event["content"]
            elif event["type"] == "thought_signature":
                thought_signature = event["content"]

        # Build content
        if text_content:
            content_items.append({"type": "text", "value": text_content})
        if thought_signature:
            content_items.append({"type": "thought_signature", "value": thought_signature})

        return {
            "role": "assistant",
            "content": content_items if len(content_items) > 1 else (text_content if text_content else ""),
        }

    @abstractmethod
    async def streaming_response(
        self,
        messages: List[UniMessage],
        model: str,
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """
        Generate content in streaming mode (stateless).

        This method should use transform_uni_config_to_model_config and
        transform_uni_message_to_model_input to prepare the request, then
        transform_model_output_to_uni_event to convert each chunk.

        Args:
            messages: List of universal message dictionaries containing conversation history
            model: Model identifier to use for generation
            config: Universal configuration dict

        Yields:
            Universal events from the streaming response
        """
        pass

    @abstractmethod
    async def streaming_response_stateful(
        self,
        message: UniMessage,
        model: str,
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """
        Generate content in streaming mode (stateful).

        This method should use transform_uni_config_to_model_config,
        transform_uni_message_to_model_input, transform_model_output_to_uni_event,
        and transform_uni_event_to_uni_message to manage the conversation flow.

        Args:
            message: Latest universal message dictionary to add to conversation
            model: Model identifier to use for generation
            config: Universal configuration dict

        Yields:
            Universal events from the streaming response
        """
        pass

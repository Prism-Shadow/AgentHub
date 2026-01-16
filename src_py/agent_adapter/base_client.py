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
        content_items = []
        text_accumulator = ""

        for event in events:
            # Merge content_items from all events
            if "content_items" in event:
                for item in event["content_items"]:
                    if item.get("type") == "text":
                        # Accumulate text items
                        text_accumulator += item.get("text", "")
                    else:
                        # Append non-text items as-is
                        content_items.append(item)

        # Add accumulated text as a single item if present
        if text_accumulator:
            content_items.insert(0, {"type": "text", "text": text_accumulator})

        return {
            "role": "assistant",
            "content_items": content_items,
        }

    @abstractmethod
    async def streaming_response(
        self,
        messages: List[UniMessage],
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """
        Generate content in streaming mode (stateless).

        This method should use transform_uni_config_to_model_config and
        transform_uni_message_to_model_input to prepare the request, then
        transform_model_output_to_uni_event to convert each chunk.

        Args:
            messages: List of universal message dictionaries containing conversation history
            config: Universal configuration dict

        Yields:
            Universal events from the streaming response
        """
        pass

    @abstractmethod
    async def streaming_response_stateful(
        self,
        message: UniMessage,
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """
        Generate content in streaming mode (stateful).

        This method should use transform_uni_config_to_model_config,
        transform_uni_message_to_model_input, transform_model_output_to_uni_event,
        and transform_uni_event_to_uni_message to manage the conversation flow.

        Args:
            message: Latest universal message dictionary to add to conversation
            config: Universal configuration dict

        Yields:
            Universal events from the streaming response
        """
        pass

    @abstractmethod
    def clear_history(self) -> None:
        """Clear the message history."""
        pass

    @abstractmethod
    def get_history(self) -> List[UniMessage]:
        """Get the current message history."""
        pass

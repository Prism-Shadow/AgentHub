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
from typing import Any, AsyncIterator

from .types import ContentItem, FinishReason, UniConfig, UniEvent, UniMessage, UsageMetadata


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All model-specific clients must inherit from this class and implement
    the required abstract methods for complete SDK abstraction.
    """

    _model: str
    _history: list[UniMessage] = []

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
    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> Any:
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

    def concat_uni_events_to_uni_message(self, events: list[UniEvent]) -> UniMessage:
        """
        Concatenate a stream of universal events into a single universal message.

        This is a concrete method implemented in the base class that can be reused
        by all model clients. It accumulates events and builds a complete message.

        Args:
            events: List of universal events from streaming response

        Returns:
            Complete universal message dictionary
        """
        content_items: list[ContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        for event in events:
            # Merge content_items from all events
            for item in event["content_items"]:
                if item["type"] == "text":
                    if (
                        content_items
                        and content_items[-1]["type"] == "text"
                        and content_items[-1].get("signature") is None
                    ):
                        content_items[-1]["text"] += item["text"]
                        if "signature" in item:  # finish the current item if signature is not None
                            content_items[-1]["signature"] = item["signature"]
                    elif item["text"]:  # omit empty text items
                        content_items.append(item.copy())
                elif item["type"] == "thinking":
                    if (
                        content_items
                        and content_items[-1]["type"] == "thinking"
                        and content_items[-1].get("signature") is None
                    ):
                        content_items[-1]["thinking"] += item["thinking"]
                        if "signature" in item:  # finish the current item if signature is not None
                            content_items[-1]["signature"] = item["signature"]
                    elif item["thinking"] or item.get("signature"):  # omit empty thinking items
                        content_items.append(item.copy())
                elif item["type"] == "partial_tool_call":
                    # Skip partial_tool_call items - they should already be converted to tool_call
                    pass
                else:
                    content_items.append(item.copy())

            usage_metadata = event.get("usage_metadata")  # usage_metadata is taken from the last event
            finish_reason = event.get("finish_reason")  # finish_reason is taken from the last event

        return {
            "role": "assistant",
            "content_items": content_items,
            "usage_metadata": usage_metadata,
            "finish_reason": finish_reason,
        }

    @abstractmethod
    async def streaming_response(
        self,
        messages: list[UniMessage],
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
        # Build a temporary messages list for inference without mutating history yet
        messages_for_inference = self._history + [message]

        # Collect all events for history
        events = []
        async for event in self.streaming_response(messages=messages_for_inference, config=config):
            events.append(event)
            yield event

        # Only update history after successful inference
        if events:
            assistant_message = self.concat_uni_events_to_uni_message(events)
            self._history.append(message)
            self._history.append(assistant_message)

        # Save history to file if trace_id is specified
        if config.get("trace_id"):
            from .integration.tracer import Tracer

            tracer = Tracer()
            tracer.save_history(self._model, self._history, config["trace_id"], config)

    @staticmethod
    def _validate_last_event(last_event: UniEvent | None) -> None:
        """Validate that the last event has usage_metadata and finish_reason.

        This validation guards against servers that silently terminate streaming
        output partway through without sending a proper final event.

        Args:
            last_event: The last event yielded by streaming_response

        Raises:
            ValueError: If last_event is None or missing usage_metadata/finish_reason
        """
        if last_event is None:
            raise ValueError("Streaming response yielded no events")

        if last_event["usage_metadata"] is None:
            raise ValueError(f"Last event must carry usage_metadata, got: {last_event}")

        if last_event["finish_reason"] is None:
            raise ValueError(f"Last event must carry finish_reason, got: {last_event}")

    def clear_history(self) -> None:
        """Clear the message history."""
        self._history.clear()

    def get_history(self) -> list[UniMessage]:
        """Get the current message history."""
        return self._history.copy()

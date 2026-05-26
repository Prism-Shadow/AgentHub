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

import base64
import json
import mimetypes
import os
import re
from typing import AsyncIterator

import httpx
from google import genai
from google.genai import types
from google.oauth2 import service_account

from ..base_client import LLMClient
from ..types import (
    EventType,
    FinishReason,
    PartialContentItem,
    PromptCaching,
    ThinkingLevel,
    ToolChoice,
    UniConfig,
    UniEvent,
    UniMessage,
    UsageMetadata,
)


class Gemini3Client(LLMClient):
    """Gemini 3-specific LLM client implementation."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize Gemini 3 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        base_url = base_url or os.getenv("GEMINI_BASE_URL")
        http_options = {"base_url": base_url} if base_url else None
        if api_key and api_key.startswith("{"):
            service_account_info = json.loads(api_key)
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._client = genai.Client(
                vertexai=True,
                credentials=credentials,
                project=service_account_info["project_id"],
                location="global",
                http_options=http_options,
            )
        else:
            self._client = genai.Client(api_key=api_key, http_options=http_options)

        self._history: list[UniMessage] = []

    def _detect_image_mime_type(self, url: str) -> str:
        """Detect MIME type from URL extension for image."""
        mime_type, _ = mimetypes.guess_type(url)
        return mime_type or "image/jpeg"

    async def _get_image_bytes_and_mime_type(self, url: str) -> dict[str, bytes | str]:
        """Get image bytes and MIME type from URL."""
        if url.startswith("data:"):
            match = re.match(r"data:([^;]+);base64,(.+)", url)
            if match:
                mime_type = match.group(1)
                base64_string = match.group(2)
                image_bytes = base64.b64decode(base64_string)
            else:
                raise ValueError(f"Invalid base64 image: {url}")
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                image_bytes = response.content
                mime_type = self._detect_image_mime_type(url)

        return {"data": image_bytes, "mime_type": mime_type}

    def _convert_thinking_level(self, thinking_level: ThinkingLevel | None) -> types.ThinkingLevel | None:
        """Convert ThinkingLevel enum to Gemini's ThinkingLevel."""
        mapping = {
            ThinkingLevel.NONE: types.ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW: types.ThinkingLevel.LOW,
            ThinkingLevel.MEDIUM: types.ThinkingLevel.MEDIUM,
            ThinkingLevel.HIGH: types.ThinkingLevel.HIGH,
            ThinkingLevel.XHIGH: types.ThinkingLevel.HIGH,
        }
        return mapping.get(thinking_level)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> types.FunctionCallingConfig:
        """Convert ToolChoice to Gemini's tool config."""
        if isinstance(tool_choice, list):
            return types.FunctionCallingConfig(mode="ANY", allowed_function_names=tool_choice)
        elif tool_choice == "none":
            return types.FunctionCallingConfig(mode="NONE")
        elif tool_choice == "auto":
            return types.FunctionCallingConfig(mode="AUTO")
        elif tool_choice == "required":
            return types.FunctionCallingConfig(mode="ANY")

    def transform_uni_config_to_model_config(self, config: UniConfig) -> types.GenerateContentConfig | None:
        """
        Transform universal configuration to Gemini-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Gemini GenerateContentConfig object or None if no config needed
        """
        config_params = {}
        if config.get("system_prompt") is not None:
            config_params["system_instruction"] = config["system_prompt"]

        if config.get("max_tokens") is not None:
            config_params["max_output_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None:
            config_params["temperature"] = config["temperature"]

        thinking_summary = config.get("thinking_summary")
        thinking_level = config.get("thinking_level")
        if thinking_summary is not None or thinking_level is not None:
            config_params["thinking_config"] = types.ThinkingConfig(
                include_thoughts=thinking_summary, thinking_level=self._convert_thinking_level(thinking_level)
            )

        if config.get("tools") is not None:
            config_params["tools"] = [types.Tool(function_declarations=config["tools"])]
            tool_choice = config.get("tool_choice")
            if tool_choice is not None:
                tool_config = self._convert_tool_choice(tool_choice)
                config_params["tool_config"] = types.ToolConfig(function_calling_config=tool_config)

        if config.get("prompt_caching") is not None and config["prompt_caching"] != PromptCaching.ENABLE:
            raise ValueError("prompt_caching must be ENABLE for Gemini 3.")

        if config.get("image_config") is not None:
            config_params["image_config"] = types.ImageConfig(**config["image_config"])

        # tts config
        if "tts" in self._model.lower():
            config_params["response_modalities"] = ["AUDIO"]
            tts_config = config.get("tts_config") or [{"voice": "Kore"}]
            if len(tts_config) not in (1, 2):
                raise ValueError("tts_config must contain 1 or 2 entries.")

            if len(tts_config) == 1:
                config_params["speech_config"] = types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=tts_config[0]["voice"])
                    )
                )
            else:
                speaker_voice_configs = []
                for speaker_config in tts_config:
                    speaker = speaker_config.get("speaker")
                    if not speaker:
                        raise ValueError("speaker is required when tts_config has 2 entries.")

                    speaker_voice_configs.append(
                        types.SpeakerVoiceConfig(
                            speaker=speaker,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=speaker_config["voice"])
                            ),
                        )
                    )

                config_params["speech_config"] = types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=speaker_voice_configs
                    )
                )

        return types.GenerateContentConfig(**config_params) if config_params else None

    async def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[types.Content]:
        """
        Transform universal message format to Gemini's Content format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of Gemini Content objects
        """
        mapping = {"user": "user", "assistant": "model"}
        contents = []
        for msg in messages:
            parts = []
            for item in msg["content_items"]:
                if item["type"] == "text":
                    parts.append(types.Part(text=item["text"], thought_signature=item.get("signature")))
                elif item["type"] == "image_url":
                    image_url = item["image_url"]
                    image_data = await self._get_image_bytes_and_mime_type(image_url)
                    parts.append(types.Part.from_bytes(**image_data))
                elif item["type"] == "inline_data":
                    inline_data = types.Blob(data=item["data"], mime_type=item["mime_type"])
                    parts.append(types.Part(inline_data=inline_data, thought_signature=item.get("signature")))
                elif item["type"] == "thinking":
                    parts.append(
                        types.Part(text=item["thinking"], thought=True, thought_signature=item.get("signature"))
                    )
                elif item["type"] == "inline_thinking":
                    inline_data = types.Blob(data=item["data"], mime_type=item["mime_type"])
                    parts.append(
                        types.Part(inline_data=inline_data, thought=True, thought_signature=item.get("signature"))
                    )
                elif item["type"] == "tool_call":
                    function_call = types.FunctionCall(name=item["name"], args=item["arguments"])
                    parts.append(types.Part(function_call=function_call, thought_signature=item.get("signature")))
                elif item["type"] == "tool_result":
                    if "tool_call_id" not in item:
                        raise ValueError("tool_call_id is required for tool result.")

                    tool_result = {"result": item["text"]}
                    multimodal_parts = []
                    if "images" in item:
                        for image_url in item["images"]:
                            image_data = await self._get_image_bytes_and_mime_type(image_url)
                            multimodal_parts.append(
                                types.FunctionResponsePart(inline_data=types.FunctionResponseBlob(**image_data))
                            )

                    parts.append(
                        types.Part.from_function_response(
                            name=item["tool_call_id"],
                            response=tool_result,
                            parts=multimodal_parts if multimodal_parts else None,
                        )
                    )
                else:
                    raise ValueError(f"Unknown item: {item}")

            contents.append(types.Content(role=mapping[msg["role"]], parts=parts))

        return contents

    def transform_model_output_to_uni_event(self, model_output: types.GenerateContentResponse) -> UniEvent:
        """
        Transform Gemini model output to universal event format.

        Args:
            model_output: Gemini response chunk

        Returns:
            Universal event dictionary
        """
        event_type: EventType = "delta"
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        if len(model_output.candidates) > 0:
            candidate = model_output.candidates[0]
            for part in candidate.content.parts:
                if part.function_call is not None:
                    content_items.append(
                        {
                            "type": "tool_call",
                            "name": part.function_call.name,
                            "arguments": part.function_call.args,
                            "tool_call_id": part.function_call.name,
                            "signature": part.thought_signature,
                        }
                    )
                elif part.thought:
                    if part.text is not None:
                        content_items.append(
                            {"type": "thinking", "thinking": part.text, "signature": part.thought_signature}
                        )
                    elif part.inline_data is not None:
                        content_items.append(
                            {
                                "type": "inline_thinking",
                                "data": part.inline_data.data,
                                "mime_type": part.inline_data.mime_type,
                                "signature": part.thought_signature,
                            }
                        )
                elif part.inline_data is not None:
                    content_items.append(
                        {
                            "type": "inline_data",
                            "data": part.inline_data.data,
                            "mime_type": part.inline_data.mime_type,
                            "signature": part.thought_signature,
                        }
                    )
                elif part.text is not None:
                    content_items.append({"type": "text", "text": part.text, "signature": part.thought_signature})
                else:
                    raise ValueError(f"Unknown output: {part}")

            if candidate.finish_reason:
                event_type = "stop"
                stop_reason_mapping = {
                    types.FinishReason.STOP: "stop",
                    types.FinishReason.MAX_TOKENS: "length",
                }
                finish_reason = stop_reason_mapping.get(candidate.finish_reason, "unknown")

        if model_output.usage_metadata:
            event_type = event_type or "delta"  # deal with separate usage data

            prompt_tokens = model_output.usage_metadata.prompt_token_count or 0
            cached_tokens = model_output.usage_metadata.cached_content_token_count or 0
            usage_metadata = {
                "cached_tokens": model_output.usage_metadata.cached_content_token_count,
                "prompt_tokens": prompt_tokens - cached_tokens,
                "thoughts_tokens": model_output.usage_metadata.thoughts_token_count,
                "response_tokens": model_output.usage_metadata.candidates_token_count,
            }

        return {
            "role": "assistant",
            "event_type": event_type,
            "content_items": content_items,
            "usage_metadata": usage_metadata,
            "finish_reason": finish_reason,
        }

    async def _embed_messages_internal(
        self,
        messages: list[UniMessage],
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        contents = await self.transform_uni_message_to_model_input(messages)
        if not contents:
            raise ValueError("Gemini embedding requires at least one content item.")

        embedding_config = config.get("embedding_config") or {}
        gemini_config = None
        if embedding_config.get("dimensions") is not None:
            gemini_config = types.EmbedContentConfig(output_dimensionality=embedding_config["dimensions"])

        result = await self._client.aio.models.embed_content(
            model=self._model,
            contents=contents,
            config=gemini_config,
        )

        yield {
            "role": "assistant",
            "event_type": "stop",
            "content_items": [
                {"type": "embedding", "embedding": list(embedding.values or [])}
                for embedding in (result.embeddings or [])
            ],
            "usage_metadata": {
                "cached_tokens": None,
                "prompt_tokens": None,
                "thoughts_tokens": None,
                "response_tokens": None,
            },
            "finish_reason": "stop",
        }

    async def _streaming_response_internal(
        self,
        messages: list[UniMessage],
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """Stream generate using Gemini SDK with unified conversion methods."""
        if "embedding" in self._model.lower():
            async for event in self._embed_messages_internal(messages, config):
                yield event
            return

        # Use unified config conversion
        gemini_config = self.transform_uni_config_to_model_config(config)

        # check if all items are text for tts model
        if "tts" in self._model.lower():
            invalid_item = next(
                (item for message in messages for item in message["content_items"] if item["type"] != "text"),
                None,
            )
            if invalid_item is not None:
                raise ValueError(f"Gemini TTS only supports text input, got content item type={invalid_item['type']}.")

        # Use unified message conversion
        contents = await self.transform_uni_message_to_model_input(messages)

        # Stream generate
        response_stream = await self._client.aio.models.generate_content_stream(
            model=self._model, contents=contents, config=gemini_config
        )
        async for chunk in response_stream:
            event = self.transform_model_output_to_uni_event(chunk)
            for item in event["content_items"]:
                if item["type"] == "tool_call":
                    # gemini 3 does not support partial tool call, mock a partial tool call event
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "partial_tool_call",
                                "name": item["name"],
                                "arguments": json.dumps(item["arguments"], ensure_ascii=False),
                                "tool_call_id": item["tool_call_id"],
                                "signature": item.get("signature"),
                            }
                        ],
                        "usage_metadata": None,
                        "finish_reason": None,
                    }

            yield event

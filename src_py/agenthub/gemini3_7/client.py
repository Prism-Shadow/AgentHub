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
from ..errors import UnsupportedParameterError
from ..types import (
    ContentItem,
    EventType,
    Fidelity,
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


class Gemini3_7Client(LLMClient):
    """Unified client for the Gemini family, named for the newest generation it serves (3.7).

    It serves every generateContent model generation (3.7 back through 3.x text, image, TTS,
    and embedding models, with the 2.5 series reachable via an explicit client_type). The API
    deprecated the temperature/top_p/top_k sampling parameters starting with the 3.6
    generation (silently ignored today, HTTP 400 in future generations), and this client
    applies that contract to the whole family: temperature is rejected everywhere.
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize Gemini 3.7 client with model and API key."""
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

    # Gemini thinking levels from weakest to strongest, used to pick the
    # closest supported level when a model rejects the requested one.
    _GEMINI_LEVEL_ORDER = (
        types.ThinkingLevel.MINIMAL,
        types.ThinkingLevel.LOW,
        types.ThinkingLevel.MEDIUM,
        types.ThinkingLevel.HIGH,
    )

    def _supported_thinking_levels(self) -> tuple[types.ThinkingLevel, ...]:
        """Thinking levels the target model accepts (llmsdk_docs/gemini3_7/docs/thinking.md).

        An empty tuple means the model rejects the thinking_level parameter
        entirely, so it must be omitted from the request.
        """
        if "gemini-2.5" in self._model:
            # The vendor table claims low/medium/high, but the live API rejects
            # every thinking_level value for the 2.5 series (verified 2026-07-24).
            return ()
        if "-image" in self._model:
            return (types.ThinkingLevel.MINIMAL, types.ThinkingLevel.HIGH)
        if "gemini-3-pro" in self._model:
            # The only pro generation without "medium".
            return (types.ThinkingLevel.LOW, types.ThinkingLevel.HIGH)
        if "-pro" in self._model:
            # Every pro generation rejects "minimal"; matching broadly keeps
            # future pro models on the safe side (clamping a level the model
            # would have accepted costs a little accuracy, forwarding an
            # unsupported one is a 400).
            return (types.ThinkingLevel.LOW, types.ThinkingLevel.MEDIUM, types.ThinkingLevel.HIGH)
        if "gemini-3.7" in self._model:
            # The 3.7 generation rejects "minimal" with a 400 (verified live 2026-08-13).
            return (types.ThinkingLevel.LOW, types.ThinkingLevel.MEDIUM, types.ThinkingLevel.HIGH)
        return self._GEMINI_LEVEL_ORDER

    def _convert_thinking_level(self, thinking_level: ThinkingLevel | None) -> types.ThinkingLevel | None:
        """Convert ThinkingLevel enum to the closest Gemini ThinkingLevel the model supports."""
        mapping = {
            ThinkingLevel.NONE: types.ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW: types.ThinkingLevel.LOW,
            ThinkingLevel.MEDIUM: types.ThinkingLevel.MEDIUM,
            ThinkingLevel.HIGH: types.ThinkingLevel.HIGH,
            ThinkingLevel.XHIGH: types.ThinkingLevel.HIGH,
            # Gemini stops at "high", so both top levels land there before per-model clamping
            ThinkingLevel.MAX: types.ThinkingLevel.HIGH,
        }
        level = mapping.get(thinking_level)
        if level is None:
            return None
        supported = self._supported_thinking_levels()
        if not supported:
            # The model takes no thinking_level at all; drop the parameter and
            # let the model use its default instead of forwarding a 400.
            return None
        if level in supported:
            return level
        # Degrade silently to the nearest supported level; ties round up,
        # e.g. MEDIUM becomes HIGH on gemini-3-pro and NONE maps to LOW on
        # gemini-3.7-flash.
        index = self._GEMINI_LEVEL_ORDER.index(level)
        return min(
            supported,
            key=lambda candidate: (
                abs(self._GEMINI_LEVEL_ORDER.index(candidate) - index),
                -self._GEMINI_LEVEL_ORDER.index(candidate),
            ),
        )

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
        Transform universal configuration to Gemini 3.7-specific configuration.

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
            raise UnsupportedParameterError(
                self.__class__.__name__,
                "temperature",
                "Gemini models do not support setting temperature; the API deprecated "
                "sampling parameters starting with the 3.6 generation.",
            )

        # include_thoughts asks for thought summaries, but whether generateContent returns any
        # is model-dependent (llmsdk_docs/gemini3_7/docs/thinking.md)
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

        if config.get("fast_mode"):
            raise UnsupportedParameterError(self.__class__.__name__, "fast_mode", "Gemini does not support fast mode.")

        if config.get("prompt_caching") is not None and config["prompt_caching"] != PromptCaching.ENABLE:
            raise UnsupportedParameterError(
                self.__class__.__name__, "prompt_caching", "prompt_caching must be ENABLE for Gemini."
            )

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

    @staticmethod
    def _part_fidelity(part: types.Part) -> dict[str, Fidelity]:
        """Wrap a part's thought signature as a fidelity payload, or nothing when absent."""
        if part.thought_signature is None:
            return {}

        return {"fidelity": {"signature": part.thought_signature}}

    @staticmethod
    def _item_thought_signature(item: ContentItem) -> str | bytes | None:
        """Read the thought signature recorded in an item's fidelity payload."""
        return (item.get("fidelity") or {}).get("signature")

    async def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[types.Content]:
        """
        Transform universal message format to Gemini's Content format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of Gemini Content objects
        """
        mapping = {"user": "user", "assistant": "model"}
        # The generateContent API wants both the call id and the function name on a function
        # response, but a universal tool_result carries only the id, so remember each call's name.
        call_names: dict[str, str] = {}
        contents = []
        for msg in messages:
            parts = []
            for item in msg["content_items"]:
                if item["type"] == "text":
                    parts.append(types.Part(text=item["text"], thought_signature=self._item_thought_signature(item)))
                elif item["type"] == "image_url":
                    image_url = item["image_url"]
                    image_data = await self._get_image_bytes_and_mime_type(image_url)
                    parts.append(types.Part.from_bytes(**image_data))
                elif item["type"] == "inline_data":
                    inline_data = types.Blob(data=item["data"], mime_type=item["mime_type"])
                    parts.append(
                        types.Part(inline_data=inline_data, thought_signature=self._item_thought_signature(item))
                    )
                elif item["type"] == "thinking":
                    parts.append(
                        types.Part(
                            text=item["thinking"], thought=True, thought_signature=self._item_thought_signature(item)
                        )
                    )
                elif item["type"] == "inline_thinking":
                    inline_data = types.Blob(data=item["data"], mime_type=item["mime_type"])
                    parts.append(
                        types.Part(
                            inline_data=inline_data, thought=True, thought_signature=self._item_thought_signature(item)
                        )
                    )
                elif item["type"] == "tool_call":
                    call_names[item["tool_call_id"]] = item["name"]
                    # Histories from before ids were stored carry the name as the tool_call_id;
                    # replay those without an id, exactly as they arrived.
                    function_call = types.FunctionCall(
                        id=item["tool_call_id"] if item["tool_call_id"] != item["name"] else None,
                        name=item["name"],
                        args=item["arguments"],
                    )
                    parts.append(
                        types.Part(function_call=function_call, thought_signature=self._item_thought_signature(item))
                    )
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

                    function_name = call_names.get(item["tool_call_id"], item["tool_call_id"])
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=item["tool_call_id"] if item["tool_call_id"] != function_name else None,
                                name=function_name,
                                response=tool_result,
                                parts=multimodal_parts if multimodal_parts else None,
                            )
                        )
                    )
                else:
                    raise ValueError(f"Unknown item: {item}")

            contents.append(types.Content(role=mapping[msg["role"]], parts=parts))

        return contents

    def transform_model_output_to_uni_event(self, model_output: types.GenerateContentResponse) -> UniEvent:
        """
        Transform Gemini 3.7 model output to universal event format.

        Args:
            model_output: Gemini response chunk

        Returns:
            Universal event dictionary
        """
        event_type: EventType = "delta"
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        if model_output.candidates:
            candidate = model_output.candidates[0]
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if part.function_call is not None:
                    content_items.append(
                        {
                            "type": "tool_call",
                            "name": part.function_call.name,
                            "arguments": part.function_call.args or {},
                            "tool_call_id": part.function_call.id or part.function_call.name,
                            **self._part_fidelity(part),
                        }
                    )
                elif part.thought:
                    if part.text is not None:
                        content_items.append({"type": "thinking", "thinking": part.text, **self._part_fidelity(part)})
                    elif part.inline_data is not None:
                        content_items.append(
                            {
                                "type": "inline_thinking",
                                "data": part.inline_data.data,
                                "mime_type": part.inline_data.mime_type,
                                **self._part_fidelity(part),
                            }
                        )
                elif part.inline_data is not None:
                    content_items.append(
                        {
                            "type": "inline_data",
                            "data": part.inline_data.data,
                            "mime_type": part.inline_data.mime_type,
                            **self._part_fidelity(part),
                        }
                    )
                elif part.text is not None:
                    content_items.append({"type": "text", "text": part.text, **self._part_fidelity(part)})
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

        if not content_items and usage_metadata is None and finish_reason is None:
            # nothing was read out of the chunk, so there is nothing to emit: a gateway
            # heartbeat looks like this, and so does any other chunk we take no value from
            event_type = "unused"

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
        """Embed transformed messages and return them as a streaming event."""
        contents = await self.transform_uni_message_to_model_input(messages)

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
                "prompt_tokens": result.metadata.billable_character_count if result.metadata else None,
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
            if event["event_type"] == "unused":
                continue

            for item in event["content_items"]:
                if item["type"] == "tool_call":
                    # gemini 3.7 does not support partial tool call, mock a partial tool call event
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "partial_tool_call",
                                "name": item["name"],
                                "arguments": json.dumps(item["arguments"], ensure_ascii=False),
                                "tool_call_id": item["tool_call_id"],
                                "fidelity": item.get("fidelity"),
                            }
                        ],
                        "usage_metadata": None,
                        "finish_reason": None,
                    }

            yield event

    async def list_models(self) -> list[str]:
        """
        List the model ids the configured endpoint serves.

        Returns:
            list[str]: The model ids, in the order the endpoint returned them.
        """
        # the API returns path-qualified names: models/gemini-3.7-flash, publishers/google/models/...
        return [model.name.split("/")[-1] async for model in await self._client.aio.models.list() if model.name]

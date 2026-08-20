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

from collections.abc import Sequence
from typing import Any

from .types import UsageMetadata


def fix_openrouter_usage_metadata(usage_metadata: UsageMetadata, base_url: str) -> UsageMetadata:
    """
    Fix the usage metadata for OpenRouter.

    OpenRouter occasionally does not include the reasoning tokens to the completion tokens.

    Args:
        usage_metadata (UsageMetadata): The usage metadata.
        base_url (str): The API URL.

    Returns:
        UsageMetadata: The fixed usage metadata.
    """
    fixed_usage_metadata = usage_metadata.copy()
    if "openrouter.ai" in base_url and fixed_usage_metadata["response_tokens"] < 0:
        fixed_usage_metadata["response_tokens"] += fixed_usage_metadata["thoughts_tokens"] or 0

    return fixed_usage_metadata


def _event_fields(model_output: Any) -> dict[str, Any]:
    """
    The event's own fields, however the SDK handed the event over.

    Args:
        model_output (Any): The stream event.

    Returns:
        dict[str, Any]: The field names and values the event carries.
    """
    if isinstance(model_output, dict):
        return model_output

    if hasattr(model_output, "model_dump"):  # a pydantic model built by a provider SDK
        # iterating keeps nested payloads as objects and avoids model_dump()'s serializer warnings
        # on the loosely built models an SDK produces for an event type it does not know
        return dict(model_output)

    return dict(vars(model_output)) if hasattr(model_output, "__dict__") else {}


def _carries_payload(value: Any) -> bool:
    """
    Whether a field value holds a non-empty structured payload rather than a scalar.

    Args:
        value (Any): The field value.

    Returns:
        bool: Whether the value holds something a client could be dropping.
    """
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return False

    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0

    return bool(_event_fields(value))


def is_foreign_no_op_event(model_output: Any, protocol_prefixes: Sequence[str]) -> bool:
    """
    Whether a stream event came from outside the protocol and carries nothing.

    Gateways in front of a model API (one-api-style proxies, OpenRouter) inject their own events into
    the SSE stream, such as heartbeats and cost tickers, and the unknown-event guard used to kill the
    whole stream on one, e.g. {"type": "ping", "cost": "@"}. Skipping is safe only where all three
    hold: the event type sits outside the protocol's own namespace, so a provider event the client has
    not learned yet (response.output_text.annotation.added, say) still raises; the type does not name
    an error, so a gateway reporting an upstream failure still raises; and no field holds a non-empty
    object or array, so an event carrying a payload the client would silently drop still raises.

    Args:
        model_output (Any): The stream event.
        protocol_prefixes (Sequence[str]): The event type prefixes the protocol owns.

    Returns:
        bool: Whether the event can be skipped.
    """
    fields = _event_fields(model_output)
    event_type = fields.get("type")
    if not isinstance(event_type, str):
        event_type = ""

    if event_type.startswith(tuple(protocol_prefixes)) or "error" in event_type or "fail" in event_type:
        return False

    return not any(_carries_payload(value) for value in fields.values())

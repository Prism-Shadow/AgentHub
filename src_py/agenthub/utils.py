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

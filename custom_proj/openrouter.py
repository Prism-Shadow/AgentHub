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

from dotenv import load_dotenv
from openai import OpenAI


def chat_with_openrouter(prompt: str) -> str:
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is not set, please add a .env file in the project root directory.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    completion = client.chat.completions.create(model="openai/gpt-4o", messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content


if __name__ == "__main__":
    result = chat_with_openrouter("Return the number of r in the word 'openrouter', output a single integer")
    print(result)

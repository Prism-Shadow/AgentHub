| We have updated our [Terms of Service](https://ai.google.dev/gemini-api/terms).

Gemini 3 is our most intelligent model family to date, built on a foundation of state-of-the-art reasoning. It is designed to bring any idea to life by mastering agentic workflows, autonomous coding, and complex multimodal tasks. This guide covers key features of the Gemini 3 model family and how to get the most out of it.  
[Try Gemini 3 Pro](https://aistudio.google.com?model=gemini-3-pro-preview) [Try Gemini 3 Flash](https://aistudio.google.com?model=gemini-3-flash-preview) [Try Nano Banana Pro](https://aistudio.google.com?model=gemini-3-pro-image-preview)

Explore our [collection of Gemini 3 apps](https://aistudio.google.com/app/apps?source=showcase&showcaseTag=gemini-3) to see how the model handles advanced reasoning, autonomous coding, and complex multimodal tasks.

Get started with a few lines of code:  

### Python

    from google import genai

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents="Find the race condition in this multi-threaded C++ snippet: [code here]",
    )

    print(response.text)

### JavaScript

    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI({});

    async function run() {
      const response = await ai.models.generateContent({
        model: "gemini-3-pro-preview",
        contents: "Find the race condition in this multi-threaded C++ snippet: [code here]",
      });

      console.log(response.text);
    }

    run();

### REST

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H 'Content-Type: application/json' \
      -X POST \
      -d '{
        "contents": [{
          "parts": [{"text": "Find the race condition in this multi-threaded C++ snippet: [code here]"}]
        }]
      }'

## Meet the Gemini 3 series

Gemini 3 Pro, the first model in the new series, is best for complex tasks that
require broad world knowledge and advanced reasoning across modalities.

Gemini 3 Flash is our latest 3-series model, with Pro-level intelligence at the
speed and pricing of Flash.

Nano Banana Pro (also known as Gemini 3 Pro Image) is our highest quality image
generation model yet.

All Gemini 3 models are currently in preview.

| Model ID | Context Window (In / Out) | Knowledge Cutoff | Pricing (Input / Output)\* |
|---|---|---|---|
| **gemini-3-pro-preview** | 1M / 64k | Jan 2025 | $2 / $12 (\<200k tokens) $4 / $18 (\>200k tokens) |
| **gemini-3-flash-preview** | 1M / 64k | Jan 2025 | $0.50 / $3 |
| **gemini-3-pro-image-preview** | 65k / 32k | Jan 2025 | $2 (Text Input) / $0.134 (Image Output)\*\* |

*\* Pricing is per 1 million tokens unless otherwise noted.*
*\*\* Image pricing varies by resolution. See the [pricing page](https://ai.google.dev/gemini-api/docs/pricing) for details.*

For detailed limits, pricing, and additional information, see the
[models page](https://ai.google.dev/gemini-api/docs/models/gemini).

## New API features in Gemini 3

Gemini 3 introduces new parameters designed to give developers more control over
latency, cost, and multimodal fidelity.

### Thinking level

Gemini 3 series models use dynamic thinking by default to reason through prompts. You can use the `thinking_level` parameter, which controls the **maximum** depth of the model's internal reasoning process before it produces a response. Gemini 3 treats these levels as relative allowances for thinking rather than strict token guarantees.

If `thinking_level` is not specified, Gemini 3 will default to `high`. For faster, lower-latency responses when complex reasoning isn't required, you can constrain the model's thinking level to `low`.

**Gemini 3 Pro and Flash thinking levels:**

The following thinking levels are supported by both Gemini 3 Pro and Flash:

- `low`: Minimizes latency and cost. Best for simple instruction following, chat, or high-throughput applications
- `high` (Default, dynamic): Maximizes reasoning depth. The model may take significantly longer to reach a first token, but the output will be more carefully reasoned.

**Gemini 3 Flash thinking levels**

In addition to the levels above, Gemini 3 Flash also supports the following
thinking levels that are not currently supported by Gemini 3 Pro:

- `minimal`: Matches the "no thinking" setting for most queries. The model may think very minimally for complex coding tasks. Minimizes latency for chat or high throughput applications.

  | **Note:** Circulation of [thought signatures](https://ai.google.dev/gemini-api/docs/gemini-3#thought_signatures) is required even when thinking level is set to `minimal` for Gemini 3 Flash.
- `medium`: Balanced thinking for most tasks.

### Python

    from google import genai
    from google.genai import types

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents="How does AI work?",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        ),
    )

    print(response.text)

### JavaScript

    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI({});

    const response = await ai.models.generateContent({
        model: "gemini-3-pro-preview",
        contents: "How does AI work?",
        config: {
          thinkingConfig: {
            thinkingLevel: "low",
          }
        },
      });

    console.log(response.text);

### REST

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H 'Content-Type: application/json' \
      -X POST \
      -d '{
        "contents": [{
          "parts": [{"text": "How does AI work?"}]
        }],
        "generationConfig": {
          "thinkingConfig": {
            "thinkingLevel": "low"
          }
        }
      }'

| **Important:** You cannot use both `thinking_level` and the legacy `thinking_budget` parameter in the same request. Doing so will return a 400 error.

### Media resolution

Gemini 3 introduces granular control over multimodal vision processing via the `media_resolution` parameter. Higher resolutions improve the model's ability to read fine text or identify small details, but increase token usage and latency. The `media_resolution` parameter determines the **maximum number of tokens allocated per input image or video frame.**

You can now set the resolution to `media_resolution_low`, `media_resolution_medium`, `media_resolution_high`, or `media_resolution_ultra_high` per individual media part or globally (via `generation_config`, global not available for ultra high). If unspecified, the model uses optimal defaults based on the media type.

**Recommended settings**

| Media Type | Recommended Setting | Max Tokens | Usage Guidance |
|---|---|---|---|
| **Images** | `media_resolution_high` | 1120 | Recommended for most image analysis tasks to ensure maximum quality. |
| **PDFs** | `media_resolution_medium` | 560 | Optimal for document understanding; quality typically saturates at `medium`. Increasing to `high` rarely improves OCR results for standard documents. |
| **Video** (General) | `media_resolution_low` (or `media_resolution_medium`) | 70 (per frame) | **Note:** For video, `low` and `medium` settings are treated identically (70 tokens) to optimize context usage. This is sufficient for most action recognition and description tasks. |
| **Video** (Text-heavy) | `media_resolution_high` | 280 (per frame) | Required only when the use case involves reading dense text (OCR) or small details within video frames. |

**Note:** The `media_resolution` parameter maps to different token counts depending on the input type. While images scale linearly (`media_resolution_low`: 280, `media_resolution_medium`: 560, `media_resolution_high`: 1120), Video is compressed more aggressively. For Video, both `media_resolution_low` and `media_resolution_medium` are capped at 70 tokens per frame, and `media_resolution_high` is capped at 280 tokens. See full details [here](https://ai.google.dev/gemini-api/docs/media-resolution#token-counts)  

### Python

    from google import genai
    from google.genai import types
    import base64

    # The media_resolution parameter is currently only available in the v1alpha API version.
    client = genai.Client(http_options={'api_version': 'v1alpha'})

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[
            types.Content(
                parts=[
                    types.Part(text="What is in this image?"),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode("..."),
                        ),
                        media_resolution={"level": "media_resolution_high"}
                    )
                ]
            )
        ]
    )

    print(response.text)

### JavaScript

    import { GoogleGenAI } from "@google/genai";

    // The media_resolution parameter is currently only available in the v1alpha API version.
    const ai = new GoogleGenAI({ apiVersion: "v1alpha" });

    async function run() {
      const response = await ai.models.generateContent({
        model: "gemini-3-pro-preview",
        contents: [
          {
            parts: [
              { text: "What is in this image?" },
              {
                inlineData: {
                  mimeType: "image/jpeg",
                  data: "...",
                },
                mediaResolution: {
                  level: "media_resolution_high"
                }
              }
            ]
          }
        ]
      });

      console.log(response.text);
    }

    run();

### REST

    curl "https://generativelanguage.googleapis.com/v1alpha/models/gemini-3-pro-preview:generateContent" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H 'Content-Type: application/json' \
      -X POST \
      -d '{
        "contents": [{
          "parts": [
            { "text": "What is in this image?" },
            {
              "inlineData": {
                "mimeType": "image/jpeg",
                "data": "..."
              },
              "mediaResolution": {
                "level": "media_resolution_high"
              }
            }
          ]
        }]
      }'

### Temperature

For Gemini 3, we strongly recommend keeping the temperature parameter at its default value of `1.0`.

While previous models often benefited from tuning temperature to control creativity versus determinism, Gemini 3's reasoning capabilities are optimized for the default setting. Changing the temperature (setting it below 1.0) may lead to unexpected behavior, such as looping or degraded performance, particularly in complex mathematical or reasoning tasks.

### Thought signatures

Gemini 3 uses [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures) to maintain reasoning context across API calls. These signatures are encrypted representations of the model's internal thought process. To ensure the model maintains its reasoning capabilities you must return these signatures back to the model in your request exactly as they were received:

- **Function Calling (Strict):** The API enforces strict validation on the "Current Turn". Missing signatures will result in a 400 error.

  | **Note:** Circulation of thought signatures is required even when [thinking level](https://ai.google.dev/gemini-api/docs/gemini-3#thinking_level) is set to `minimal` for Gemini 3 Flash.
- **Text/Chat:** Validation is not strictly enforced, but omitting signatures will degrade the model's reasoning and answer quality.

- **Image generation/editing (Strict)** : The API enforces strict validation on all Model parts including a `thoughtSignature`. Missing signatures will result in a 400 error.

| **Success:** If you use the [official SDKs (Python, Node, Java)](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#thinking) and standard chat history, Thought Signatures are handled automatically. You do not need to manually manage these fields.

#### Function calling (strict validation)

When Gemini generates a `functionCall`, it relies on the `thoughtSignature` to process the tool's output correctly in the next turn. The "Current Turn" includes all Model (`functionCall`) and User (`functionResponse`) steps that occurred since the last standard **User** `text` message.

- **Single Function Call:** The `functionCall` part contains a signature. You must return it.
- **Parallel Function Calls:** Only the first `functionCall` part in the list will contain the signature. You must return the parts in the exact order received.
- **Multi-Step (Sequential):** If the model calls a tool, receives a result, and calls *another* tool (within the same turn), **both** function calls have signatures. You must return **all** accumulated signatures in the history.

#### Text and streaming

For standard chat or text generation, the presence of a signature is not guaranteed.

- **Non-Streaming** : The final content part of the response may contain a `thoughtSignature`, though it is not always present. If one is returned, you should send it back to maintain best performance.
- **Streaming**: If a signature is generated, it may arrive in a final chunk that contains an empty text part. Ensure your stream parser checks for signatures even if the text field is empty.

#### Image generation and editing

For `gemini-3-pro-image-preview`, thought signatures are critical for conversational editing. When you ask the model to modify an image it relies on the `thoughtSignature` from the previous turn to understand the composition and logic of the original image.

- **Editing:** Signatures are guaranteed on the first part after the thoughts of the response (`text` or `inlineData`) and on every subsequent `inlineData` part. You must return all of these signatures to avoid errors.

#### Code examples

#### Multi-step Function Calling (Sequential)

The user asks a question requiring two separate steps (Check Flight -\> Book Taxi) in one turn.  


**Step 1: Model calls Flight Tool.**  

The model returns a signature `<Sig_A>`  

```java
// Model Response (Turn 1, Step 1)
  {
    "role": "model",
    "parts": [
      {
        "functionCall": { "name": "check_flight", "args": {...} },
        "thoughtSignature": "<Sig_A>" // SAVE THIS
      }
    ]
  }
```

**Step 2: User sends Flight Result**  

We must send back `<Sig_A>` to keep the model's train of thought.  

```java
// User Request (Turn 1, Step 2)
[
  { "role": "user", "parts": [{ "text": "Check flight AA100..." }] },
  {
    "role": "model",
    "parts": [
      {
        "functionCall": { "name": "check_flight", "args": {...} },
        "thoughtSignature": "<Sig_A>" // REQUIRED
      }
    ]
  },
  { "role": "user", "parts": [{ "functionResponse": { "name": "check_flight", "response": {...} } }] }
]
```

**Step 3: Model calls Taxi Tool**  

The model remembers the flight delay via `<Sig_A>` and now decides to book a taxi. It generates a *new* signature `<Sig_B>`.  

```java
// Model Response (Turn 1, Step 3)
{
  "role": "model",
  "parts": [
    {
      "functionCall": { "name": "book_taxi", "args": {...} },
      "thoughtSignature": "<Sig_B>" // SAVE THIS
    }
  ]
}
```

**Step 4: User sends Taxi Result**  

To complete the turn, you must send back the entire chain: `<Sig_A>` AND `<Sig_B>`.  

```java
// User Request (Turn 1, Step 4)
[
  // ... previous history ...
  {
    "role": "model",
    "parts": [
       { "functionCall": { "name": "check_flight", ... }, "thoughtSignature": "<Sig_A>" }
    ]
  },
  { "role": "user", "parts": [{ "functionResponse": {...} }] },
  {
    "role": "model",
    "parts": [
       { "functionCall": { "name": "book_taxi", ... }, "thoughtSignature": "<Sig_B>" }
    ]
  },
  { "role": "user", "parts": [{ "functionResponse": {...} }] }
]
```  

#### Parallel Function Calling

The user asks: "Check the weather in Paris and London." The model returns two function calls in one response.  

```java
// User Request (Sending Parallel Results)
[
  {
    "role": "user",
    "parts": [
      { "text": "Check the weather in Paris and London." }
    ]
  },
  {
    "role": "model",
    "parts": [
      // 1. First Function Call has the signature
      {
        "functionCall": { "name": "check_weather", "args": { "city": "Paris" } },
        "thoughtSignature": "<Signature_A>"
      },
      // 2. Subsequent parallel calls DO NOT have signatures
      {
        "functionCall": { "name": "check_weather", "args": { "city": "London" } }
      }
    ]
  },
  {
    "role": "user",
    "parts": [
      // 3. Function Responses are grouped together in the next block
      {
        "functionResponse": { "name": "check_weather", "response": { "temp": "15C" } }
      },
      {
        "functionResponse": { "name": "check_weather", "response": { "temp": "12C" } }
      }
    ]
  }
]
```  

#### Text/In-Context Reasoning (No Validation)

The user asks a question that requires in-context reasoning without external tools. While not strictly validated, including the signature helps the model maintain the reasoning chain for follow-up questions.  

```java
// User Request (Follow-up question)
[
  {
    "role": "user",
    "parts": [{ "text": "What are the risks of this investment?" }]
  },
  {
    "role": "model",
    "parts": [
      {
        "text": "I need to calculate the risk step-by-step. First, I'll look at volatility...",
        "thoughtSignature": "<Signature_C>" // Recommended to include
      }
    ]
  },
  {
    "role": "user",
    "parts": [{ "text": "Summarize that in one sentence." }]
  }
]
```  

#### Image Generation \& Editing

For image generation, signatures are strictly validated. They appear on the **first part** (text or image) and **all subsequent image parts**. All must be returned in the next turn.  

```java
// Model Response (Turn 1)
{
  "role": "model",
  "parts": [
    // 1. First part ALWAYS has a signature (even if text)
    {
      "text": "I will generate a cyberpunk city...",
      "thoughtSignature": "<Signature_D>"
    },
    // 2. ALL InlineData (Image) parts ALWAYS have signatures
    {
      "inlineData": { ... },
      "thoughtSignature": "<Signature_E>"
    },
  ]
}

// User Request (Turn 2 - Requesting an Edit)
{
  "contents": [
    // History must include ALL signatures received
    {
      "role": "user",
      "parts": [{ "text": "Generate a cyberpunk city" }]
    },
    {
      "role": "model",
      "parts": [
         { "text": "...", "thoughtSignature": "<Signature_D>" },
         { "inlineData": "...", "thoughtSignature": "<Signature_E>" },
      ]
    },
    // New User Prompt
    {
      "role": "user",
      "parts": [{ "text": "Make it daytime." }]
    }
  ]
}
```

#### Migrating from other models

If you are transferring a conversation trace from another model (e.g., Gemini 2.5) or injecting a custom function call that was not generated by Gemini 3, you will not have a valid signature.

To bypass strict validation in these specific scenarios, populate the field with this specific dummy string: `"thoughtSignature": "context_engineering_is_the_way_to_go"`

### Structured Outputs with tools

Gemini 3 models allow you to combine [Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output) with built-in tools, including [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search), [URL Context](https://ai.google.dev/gemini-api/docs/url-context), and [Code Execution](https://ai.google.dev/gemini-api/docs/code-execution).  

### Python

    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
    from typing import List

    class MatchResult(BaseModel):
        winner: str = Field(description="The name of the winner.")
        final_match_score: str = Field(description="The final match score.")
        scorers: List[str] = Field(description="The name of the scorer.")

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents="Search for all details for the latest Euro.",
        config={
            "tools": [
                {"google_search": {}},
                {"url_context": {}}
            ],
            "response_mime_type": "application/json",
            "response_json_schema": MatchResult.model_json_schema(),
        },  
    )

    result = MatchResult.model_validate_json(response.text)
    print(result)

### JavaScript

    import { GoogleGenAI } from "@google/genai";
    import { z } from "zod";
    import { zodToJsonSchema } from "zod-to-json-schema";

    const ai = new GoogleGenAI({});

    const matchSchema = z.object({
      winner: z.string().describe("The name of the winner."),
      final_match_score: z.string().describe("The final score."),
      scorers: z.array(z.string()).describe("The name of the scorer.")
    });

    async function run() {
      const response = await ai.models.generateContent({
        model: "gemini-3-pro-preview",
        contents: "Search for all details for the latest Euro.",
        config: {
          tools: [
            { googleSearch: {} },
            { urlContext: {} }
          ],
          responseMimeType: "application/json",
          responseJsonSchema: zodToJsonSchema(matchSchema),
        },
      });

      const match = matchSchema.parse(JSON.parse(response.text));
      console.log(match);
    }

    run();

### REST

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H 'Content-Type: application/json' \
      -X POST \
      -d '{
        "contents": [{
          "parts": [{"text": "Search for all details for the latest Euro."}]
        }],
        "tools": [
          {"googleSearch": {}},
          {"urlContext": {}}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "winner": {"type": "string", "description": "The name of the winner."},
                    "final_match_score": {"type": "string", "description": "The final score."},
                    "scorers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The name of the scorer."
                    }
                },
                "required": ["winner", "final_match_score", "scorers"]
            }
        }
      }'

### Image generation

Gemini 3 Pro Image lets you generate and edit images from text prompts. It uses reasoning to "think" through a prompt and can retrieve real-time data---such as weather forecasts or stock charts---before using [Google Search](https://ai.google.dev/gemini-api/docs/google-search) grounding before generating high-fidelity images.

**New \& improved capabilities:**

- **4K \& text rendering:** Generate sharp, legible text and diagrams with up to 2K and 4K resolutions.
- **Grounded generation:** Use the `google_search` tool to verify facts and generate imagery based on real-world information.
- **Conversational editing:** Multi-turn image editing by simply asking for changes (e.g., "Make the background a sunset"). This workflow relies on **Thought Signatures** to preserve visual context between turns.

For complete details on aspect ratios, editing workflows, and configuration options, see the [Image Generation guide](https://ai.google.dev/gemini-api/docs/image-generation).  

### Python

    from google import genai
    from google.genai import types

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents="Generate an infographic of the current weather in Tokyo.",
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                image_size="4K"
            )
        )
    )

    image_parts = [part for part in response.parts if part.inline_data]

    if image_parts:
        image = image_parts[0].as_image()
        image.save('weather_tokyo.png')
        image.show()

### JavaScript

    import { GoogleGenAI } from "@google/genai";
    import * as fs from "node:fs";

    const ai = new GoogleGenAI({});

    async function run() {
      const response = await ai.models.generateContent({
        model: "gemini-3-pro-image-preview",
        contents: "Generate a visualization of the current weather in Tokyo.",
        config: {
          tools: [{ googleSearch: {} }],
          imageConfig: {
            aspectRatio: "16:9",
            imageSize: "4K"
          }
        }
      });

      for (const part of response.candidates[0].content.parts) {
        if (part.inlineData) {
          const imageData = part.inlineData.data;
          const buffer = Buffer.from(imageData, "base64");
          fs.writeFileSync("weather_tokyo.png", buffer);
        }
      }
    }

    run();

### REST

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H 'Content-Type: application/json' \
      -X POST \
      -d '{
        "contents": [{
          "parts": [{"text": "Generate a visualization of the current weather in Tokyo."}]
        }],
        "tools": [{"googleSearch": {}}],
        "generationConfig": {
            "imageConfig": {
              "aspectRatio": "16:9",
              "imageSize": "4K"
          }
        }
      }'

**Example Response**

![Weather Tokyo](https://ai.google.dev/static/gemini-api/docs/images/weather-tokyo.jpg)

### Multimodal function responses

[Multimodal function calling](https://ai.google.dev/gemini-api/docs/function-calling#multimodal)
allows users to have function responses containing
multimodal objects allowing for improved utilization of function calling
capabilities of the model. Standard function calling only supports text-based
function responses:  

### Python

    from google import genai
    from google.genai import types

    import requests

    client = genai.Client()

    # This is a manual, two turn multimodal function calling workflow:

    # 1. Define the function tool
    get_image_declaration = types.FunctionDeclaration(
      name="get_image",
      description="Retrieves the image file reference for a specific order item.",
      parameters={
          "type": "object",
          "properties": {
              "item_name": {
                  "type": "string",
                  "description": "The name or description of the item ordered (e.g., 'instrument')."
              }
          },
          "required": ["item_name"],
      },
    )
    tool_config = types.Tool(function_declarations=[get_image_declaration])

    # 2. Send a message that triggers the tool
    prompt = "Show me the instrument I ordered last month."
    response_1 = client.models.generate_content(
      model="gemini-3-flash-preview",
      contents=[prompt],
      config=types.GenerateContentConfig(
          tools=[tool_config],
      )
    )

    # 3. Handle the function call
    function_call = response_1.function_calls[0]
    requested_item = function_call.args["item_name"]
    print(f"Model wants to call: {function_call.name}")

    # Execute your tool (e.g., call an API)
    # (This is a mock response for the example)
    print(f"Calling external tool for: {requested_item}")

    function_response_data = {
      "image_ref": {"$ref": "instrument.jpg"},
    }
    image_path = "https://goo.gle/instrument-img"
    image_bytes = requests.get(image_path).content
    function_response_multimodal_data = types.FunctionResponsePart(
      inline_data=types.FunctionResponseBlob(
        mime_type="image/jpeg",
        display_name="instrument.jpg",
        data=image_bytes,
      )
    )

    # 4. Send the tool's result back
    # Append this turn's messages to history for a final response.
    history = [
      types.Content(role="user", parts=[types.Part(text=prompt)]),
      response_1.candidates[0].content,
      types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
              name=function_call.name,
              response=function_response_data,
              parts=[function_response_multimodal_data]
            )
        ],
      )
    ]

    response_2 = client.models.generate_content(
      model="gemini-3-flash-preview",
      contents=history,
      config=types.GenerateContentConfig(
          tools=[tool_config],
          thinking_config=types.ThinkingConfig(include_thoughts=True)
      ),
    )

    print(f"\nFinal model response: {response_2.text}")

### JavaScript

    import { GoogleGenAI, Type } from '@google/genai';

    const client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

    // This is a manual, two turn multimodal function calling workflow:
    // 1. Define the function tool
    const getImageDeclaration = {
      name: 'get_image',
      description: 'Retrieves the image file reference for a specific order item.',
      parameters: {
        type: Type.OBJECT,
        properties: {
          item_name: {
            type: Type.STRING,
            description: "The name or description of the item ordered (e.g., 'instrument').",
          },
        },
        required: ['item_name'],
      },
    };

    const toolConfig = {
      functionDeclarations: [getImageDeclaration],
    };

    // 2. Send a message that triggers the tool
    const prompt = 'Show me the instrument I ordered last month.';
    const response1 = await client.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
      config: {
        tools: [toolConfig],
      },
    });

    // 3. Handle the function call
    const functionCall = response1.functionCalls[0];
    const requestedItem = functionCall.args.item_name;
    console.log(`Model wants to call: ${functionCall.name}`);

    // Execute your tool (e.g., call an API)
    // (This is a mock response for the example)
    console.log(`Calling external tool for: ${requestedItem}`);

    const functionResponseData = {
      image_ref: { $ref: 'instrument.jpg' },
    };

    const imageUrl = "https://goo.gle/instrument-img";
    const response = await fetch(imageUrl);
    const imageArrayBuffer = await response.arrayBuffer();
    const base64ImageData = Buffer.from(imageArrayBuffer).toString('base64');

    const functionResponseMultimodalData = {
      inlineData: {
        mimeType: 'image/jpeg',
        displayName: 'instrument.jpg',
        data: base64ImageData,
      },
    };

    // 4. Send the tool's result back
    // Append this turn's messages to history for a final response.
    const history = [
      { role: 'user', parts: [{ text: prompt }] },
      response1.candidates[0].content,
      {
        role: 'tool',
        parts: [
          {
            functionResponse: {
              name: functionCall.name,
              response: functionResponseData,
              parts: [functionResponseMultimodalData],
            },
          },
        ],
      },
    ];

    const response2 = await client.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: history,
      config: {
        tools: [toolConfig],
        thinkingConfig: { includeThoughts: true },
      },
    });

    console.log(`\nFinal model response: ${response2.text}`);

### REST

    IMG_URL="https://goo.gle/instrument-img"

    MIME_TYPE=$(curl -sIL "$IMG_URL" | grep -i '^content-type:' | awk -F ': ' '{print $2}' | sed 's/\r$//' | head -n 1)
    if [[ -z "$MIME_TYPE" || ! "$MIME_TYPE" == image/* ]]; then
      MIME_TYPE="image/jpeg"
    fi

    # Check for macOS
    if [[ "$(uname)" == "Darwin" ]]; then
      IMAGE_B64=$(curl -sL "$IMG_URL" | base64 -b 0)
    elif [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
      IMAGE_B64=$(curl -sL "$IMG_URL" | base64)
    else
      IMAGE_B64=$(curl -sL "$IMG_URL" | base64 -w0)
    fi

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H 'Content-Type: application/json' \
      -X POST \
      -d '{
        "contents": [
          ...,
          {
            "role": "user",
            "parts": [
            {
                "functionResponse": {
                  "name": "get_image",
                  "response": {
                    "image_ref": {
                      "$ref": "instrument.jpg"
                    }
                  },
                  "parts": [
                    {
                      "inlineData": {
                        "displayName": "instrument.jpg",
                        "mimeType":"'"$MIME_TYPE"'",
                        "data": "'"$IMAGE_B64"'"
                      }
                    }
                  ]
                }
              }
            ]
          }
        ]
      }'

## Migrating from Gemini 2.5

Gemini 3 is our most capable model family to date and offers a stepwise improvement over Gemini 2.5. When migrating, consider the following:

- **Thinking:** If you were previously using complex prompt engineering (like chain of thought) to force Gemini 2.5 to reason, try Gemini 3 with `thinking_level: "high"` and simplified prompts.
- **Temperature settings:** If your existing code explicitly sets temperature (especially to low values for deterministic outputs), we recommend removing this parameter and using the Gemini 3 default of 1.0 to avoid potential looping issues or performance degradation on complex tasks.
- **PDF \& document understanding:** Default OCR resolution for PDFs has changed. If you relied on specific behavior for dense document parsing, test the new `media_resolution_high` setting to ensure continued accuracy.
- **Token consumption:** Migrating to Gemini 3 defaults may **increase** token usage for PDFs but **decrease** token usage for video. If requests now exceed the context window due to higher default resolutions, we recommend explicitly reducing the media resolution.
- **Image segmentation:** Image segmentation capabilities (returning pixel-level masks for objects) are not supported in Gemini 3 Pro or Gemini 3 Flash. For workloads requiring native image segmentation, we recommend continuing to utilize Gemini 2.5 Flash with thinking turned off or [Gemini Robotics-ER 1.5](https://ai.google.dev/gemini-api/docs/robotics-overview).
- **Tool support**: Maps grounding and Computer use tools are not yet supported for Gemini 3 models, so won't migrate. Additionally, combining built-in tools with function calling is not yet supported.

## OpenAI compatibility

For users utilizing the OpenAI compatibility layer, standard parameters are automatically mapped to Gemini equivalents:

- `reasoning_effort` (OAI) maps to `thinking_level` (Gemini). Note that `reasoning_effort` medium maps to `thinking_level` high on Gemini 3 Flash.

## Prompting best practices

Gemini 3 is a reasoning model, which changes how you should prompt.

- **Precise instructions:** Be concise in your input prompts. Gemini 3 responds best to direct, clear instructions. It may over-analyze verbose or overly complex prompt engineering techniques used for older models.
- **Output verbosity:** By default, Gemini 3 is less verbose and prefers providing direct, efficient answers. If your use case requires a more conversational or "chatty" persona, you must explicitly steer the model in the prompt (e.g., "Explain this as a friendly, talkative assistant").
- **Context management:** When working with large datasets (e.g., entire books, codebases, or long videos), place your specific instructions or questions at the end of the prompt, after the data context. Anchor the model's reasoning to the provided data by starting your question with a phrase like, "Based on the information above...".

Learn more about prompt design strategies in the [prompt engineering guide](https://ai.google.dev/gemini-api/docs/prompting-strategies).

## FAQ

1. **What is the knowledge cutoff for Gemini 3?** Gemini 3 models have a knowledge cutoff of January 2025. For more recent information, use the [Search Grounding](https://ai.google.dev/gemini-api/docs/google-search) tool.

2. **What are the context window limits?** Gemini 3 models support a 1 million token input context window and up to 64k tokens of output.

3. **Is there a free tier for Gemini 3?** Gemini 3 Flash `gemini-3-flash-preview` has a free tier in the Gemini API. You can try both Gemini 3 Pro and Flash for free in Google AI Studio, but currently, there is no free tier available for `gemini-3-pro-preview` in the Gemini API.

4. **Will my old `thinking_budget` code still work?** Yes, `thinking_budget` is still supported for backward compatibility, but we recommend migrating to `thinking_level` for more predictable performance. Do not use both in the same request.

5. **Does Gemini 3 support the Batch API?** Yes, Gemini 3 supports the [Batch API](https://ai.google.dev/gemini-api/docs/batch-api).

6. **Is Context Caching supported?** Yes, [Context Caching](https://ai.google.dev/gemini-api/docs/caching) is supported for Gemini 3.

7. **Which tools are supported in Gemini 3?** Gemini 3 supports [Google Search](https://ai.google.dev/gemini-api/docs/google-search), [File Search](https://ai.google.dev/gemini-api/docs/file-search), [Code Execution](https://ai.google.dev/gemini-api/docs/code-execution), and [URL Context](https://ai.google.dev/gemini-api/docs/url-context). It also supports standard [Function Calling](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting) for your own custom tools (but not with built-in tools). Please note that [Grounding with Google Maps](https://ai.google.dev/gemini-api/docs/maps-grounding) and [Computer Use](https://ai.google.dev/gemini-api/docs/computer-use) are currently not supported.

   | **Note:** Gemini 3 billing for [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search) will begin on January 5, 2026.

## Next steps

- Get started with the [Gemini 3 Cookbook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb#templateParams=%7B%22MODEL_ID%22%3A+%22gemini-3-pro-preview%22%7D)
- Check the dedicated Cookbook guide on [thinking levels](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started_thinking_REST.ipynb#gemini3) and how to migrate from thinking budget to thinking levels.

> ## Documentation Index
> Fetch the complete documentation index at: https://platform.minimax.io/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# API Overview

> Overview of MiniMax API capabilities including language, speech, video, image, music, and file management.

## Get API Key

* **Pay-as-you-go**：Visit [API Keys > Create new secret key](https://platform.minimax.io/user-center/basic-information/interface-key) to get your **API Key**
  <Note>Pay-as-you-go supports all modality models, including language, Video, Speech, and Image.</Note>

* **Token Plan**：Visit [Billing > Token Plan](https://platform.minimax.io/user-center/payment/token-plan) to view your **Subscription Key**
  <Note>The Subscription Key is used for Token Plan subscriptions and purchased Credits. It is separate from pay-as-you-go API Keys. See [Token Plan Overview](https://platform.minimax.io/docs/token-plan/intro) for details.</Note>

***

## LLM

The LLM API uses **MiniMax M3**, **MiniMax M2.7**, **MiniMax M2.7 highspeed**, **MiniMax M2.5**, **MiniMax M2.5 highspeed**, **MiniMax M2.1**, **MiniMax M2.1 highspeed**, and **MiniMax M2** to generate conversational content and trigger tool calls based on the provided context.

It can be accessed via **HTTP requests**, the **Anthropic SDK** (Recommended), or the **OpenAI SDK**.

**Supported Models**

| Model Name             | Context Window | Description                                                                                                                                   |
| :--------------------- | :------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| MiniMax-M3             | 1,000,000      | **Latest M-series language model for agentic reasoning, tool use, coding, and long-context tasks**                                            |
| MiniMax-M2.7           | 204,800        | **Beginning the journey of recursive self-improvement. (output speed approximately 60 tps)**                                                  |
| MiniMax-M2.7-highspeed | 204,800        | **M2.7 highspeed: Same performance, faster and more agile (output speed approximately 100 tps)**                                              |
| MiniMax-M2.5           | 204,800        | **Peak Performance. Ultimate Value. Master the Complex (output speed approximately 60 tps)**                                                  |
| MiniMax-M2.5-highspeed | 204,800        | **M2.5 highspeed: Same performance, faster and more agile (output speed approximately 100 tps)**                                              |
| MiniMax-M2.1           | 204,800        | **Powerful Multi-Language Programming Capabilities with Comprehensively Enhanced Programming Experience (output speed approximately 60 tps)** |
| MiniMax-M2.1-highspeed | 204,800        | **Faster and More Agile (output speed approximately 100 tps)**                                                                                |
| MiniMax-M2             | 204,800        | **Agentic capabilities, Advanced reasoning**                                                                                                  |

Please note: The maximum token count refers to the total number of input and output tokens.

<Columns cols={2}>
  <Card title="Anthropic API Compatible (Recommended)" icon="book-open" href="/docs/api-reference/text-anthropic-api" cta="View Docs">
    Use Anthropic SDK with MiniMax models
  </Card>

  <Card title="OpenAI API Compatible" icon="book-open" href="/docs/api-reference/text-openai-api" cta="View Docs">
    Use OpenAI SDK with MiniMax models
  </Card>
</Columns>

***

## MiniMax-H3 \<NEW>

This API is powered by MiniMax-H3 and supports video generation from multimodal input (text, images, video, audio), covering text-to-video, image-to-video, first-and-last-frame, and reference-to-video scenarios.

**Supported Models**

| Model      | Description                                                                                                                               |
| :--------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| MiniMax-H3 | Multimodal video generation model supporting text / image / first-and-last-frame / reference input, 768P / 2K resolution, 4–15s duration. |

**API Usage Guide**

MiniMax-H3 tasks are asynchronous. There are three creation endpoints—**Create Video Generation Task**, **Create H3-Context-IR Task**, and **Create Video Regeneration Task**—and shared endpoints for querying, listing, and cancelling or deleting tasks. The workflow is as follows:

1. Create a video generation task, create an H3-Context-IR task with the same multimodal input, or create a video regeneration task for a source video that meets the MiniMax-H3 768P output specifications. A regeneration request must contain exactly one source-video item with `role=base_video`. All three endpoints return a `task_id` on success.
2. Use **Query Task** with the `task_id` to retrieve its status and result. When a video task succeeds, get its output URL from `content.url`; when an H3-Context-IR task succeeds, get the enhanced prompt from `content.prompt`. You can also use **List Tasks** and distinguish `generation`, `h3_context_ir`, and `regeneration` with `task_type`.
3. Use **Cancel or Delete Task** to cancel a queued task or delete a succeeded or failed task record.

<Columns cols={2}>
  <Card title="Create Video Generation Task" icon="circle-play" href="/docs/api-reference/video-generation-v2-create" cta="View Docs">
    Create a video generation task from multimodal content input
  </Card>

  <Card title="Create H3-Context-IR Task" icon="pen-to-square" href="/docs/api-reference/video-generation-v2-h3-context-ir" cta="View Docs">
    Deeply interpret multimodal video-generation context and produce a structured, enhanced prompt
  </Card>

  <Card title="Create Video Regeneration Task" icon="wand-magic-sparkles" href="/docs/api-reference/video-generation-v2-regeneration" cta="View Docs">
    Regenerate a video that meets the MiniMax-H3 768P output specifications as a 2K video
  </Card>

  <Card title="Query Task" icon="search" href="/docs/api-reference/video-generation-v2-query" cta="View Docs">
    Query task status by task\_id and get the video download URL
  </Card>

  <Card title="List Tasks" icon="list" href="/docs/api-reference/video-generation-v2-list" cta="View Docs">
    List tasks from the last 7 days and filter by task type
  </Card>

  <Card title="Cancel or Delete Task" icon="trash" href="/docs/api-reference/video-generation-v2-delete" cta="View Docs">
    Cancel a queued task or delete a succeeded or failed task record
  </Card>
</Columns>

***

## Text to Speech

This API provides synchronous text-to-speech (T2A) generation, supporting up to **10,000** characters per request.
The interface is stateless: each call only processes the provided input without involving business logic, and the model does not store any user data.

**Key Features**

1. Access to 300+ system voices and custom cloned voices.
2. Adjustable volume, pitch, speed, and output formats.
3. Support for proportional audio mixing.
4. Configurable fixed time intervals.
5. Multiple audio formats and specifications supported: `mp3`, `pcm`, `flac`, `wav`.
6. Support for streaming output.

**Typical Use Cases:** short text generation, voice chat, online social interactions.

**Supported Models**

| Model            | Description                                                                                              |
| :--------------- | :------------------------------------------------------------------------------------------------------- |
| speech-2.8-hd    | Latest HD model. Ultra-realistic quality featuring sound tags.                                           |
| speech-2.8-turbo | Latest Turbo model. Seamless speed meets natural flow.                                                   |
| speech-2.6-hd    | HD model with outstanding prosody and excellent cloning similarity.                                      |
| speech-2.6-turbo | Turbo model with support for 40 languages.                                                               |
| speech-02-hd     | Superior rhythm and stability, with outstanding performance in replication similarity and sound quality. |
| speech-02-turbo  | Superior rhythm and stability, with enhanced multilingual capabilities and excellent performance.        |

**Available Interfaces**

Synchronous speech synthesis provides two interfaces. Choose based on your needs:

* HTTP T2A API
* WebSocket T2A API

### Supported Languages

MiniMax speech synthesis models offer robust multilingual capability, supporting **40 widely used languages** worldwide.

| Support Languages |               |               |
| ----------------- | ------------- | ------------- |
| 1. Chinese        | 15. Turkish   | 28. Malay     |
| 2. Cantonese      | 16. Dutch     | 29. Persian   |
| 3. English        | 17. Ukrainian | 30. Slovak    |
| 4. Spanish        | 18. Thai      | 31. Swedish   |
| 5. French         | 19. Polish    | 32. Croatian  |
| 6. Russian        | 20. Romanian  | 33. Filipino  |
| 7. German         | 21. Greek     | 34. Hungarian |
| 8. Portuguese     | 22. Czech     | 35. Norwegian |
| 9. Arabic         | 23. Finnish   | 36. Slovenian |
| 10. Italian       | 24. Hindi     | 37. Catalan   |
| 11. Japanese      | 25. Bulgarian | 38. Nynorsk   |
| 12. Korean        | 26. Danish    | 39. Tamil     |
| 13. Indonesian    | 27. Hebrew    | 40. Afrikaans |
| 14. Vietnamese    |               |               |

<Columns cols={2}>
  <Card title="HTTP T2A API" icon="globe" href="/docs/api-reference/speech-t2a-http" cta="View Docs">
    Synchronous speech synthesis via HTTP
  </Card>

  <Card title="WebSocket T2A API" icon="plug" href="/docs/api-reference/speech-t2a-websocket" cta="View Docs">
    Streaming speech synthesis via WebSocket
  </Card>
</Columns>

***

## Asynchronous Long-Text Speech Generation

This API supports asynchronous text-to-speech generation. Each request can handle up to **1 million characters**, and the resulting audio can be retrieved asynchronously.

Features supported:

1. Choose from 100+ system voices and cloned voices.
2. Customize pitch, speed, volume, bitrate, sample rate, and output format.
3. Retrieve audio metadata, such as duration and file size.
4. Retrieve precise sentence-level timestamps (subtitles).
5. Input text directly as a string or via `file_id` after uploading a text file.
6. Detect illegal characters:
   * If illegal characters are **≤10%**, audio is generated normally, with the ratio returned.
   * If illegal characters are **>10%**, no audio will be generated (an error code will be returned).

**Note:** The returned audio URL is valid for **9 hours** (32,400 seconds) from the time it is issued. After expiration, the URL becomes invalid and the generated data will be lost.

**Use Case:** Converting entire books or other long texts into audio.

**Supported Models**

| Model            | Description                                                                                              |
| :--------------- | :------------------------------------------------------------------------------------------------------- |
| speech-2.8-hd    | Latest HD model. Ultra-realistic quality featuring sound tags.                                           |
| speech-2.8-turbo | Latest Turbo model. Seamless speed meets natural flow.                                                   |
| speech-2.6-hd    | HD model with outstanding prosody and excellent cloning similarity.                                      |
| speech-2.6-turbo | Turbo model with support for 40 languages.                                                               |
| speech-02-hd     | Superior rhythm and stability, with outstanding performance in replication similarity and sound quality. |
| speech-02-turbo  | Superior rhythm and stability, with enhanced multilingual capabilities and excellent performance.        |

**API Overview**

This feature includes **two APIs**:

1. Create a speech generation task (returns `task_id`).
2. Query the speech generation task status using `task_id`.
3. If the task succeeds, use the returned `file_id` with the **File API** to view and download the result.

<Columns cols={2}>
  <Card title="Create Async Task" icon="circle-play" href="/docs/api-reference/speech-t2a-async-create" cta="View Docs">
    Create a long-text speech generation task
  </Card>

  <Card title="Query Task Status" icon="search" href="/docs/api-reference/speech-t2a-async-query" cta="View Docs">
    Query speech generation task status
  </Card>
</Columns>

***

## Voice Cloning

This API supports cloning voices from user-uploaded audio files along with optional sample audio to enhance cloning quality.

**Use cases:** fast replication of a target timbre (IP voice recreation, voice cloning) where you need to quickly clone a specific voice.

The API supports cloning from mono or stereo audio and can rapidly reproduce speech that matches the timbre of a provided reference file.

**Supported Models**

| Model            | Description                                                                                              |
| :--------------- | :------------------------------------------------------------------------------------------------------- |
| speech-2.8-hd    | Latest HD model. Ultra-realistic quality featuring sound tags.                                           |
| speech-2.8-turbo | Latest Turbo model. Seamless speed meets natural flow.                                                   |
| speech-2.6-hd    | HD model with real-time response, intelligent parsing, fluent LoRA voice                                 |
| speech-2.6-turbo | Turbo model. Ultimate Value, 40 Languages                                                                |
| speech-02-hd     | Superior rhythm and stability, with outstanding performance in replication similarity and sound quality. |
| speech-02-turbo  | Superior rhythm and stability, with enhanced multilingual capabilities and excellent performance.        |

### Notes

* Using this API to clone a voice **does not** immediately incur a cloning fee. The cloning fee is charged the **first time** you synthesize speech with the cloned voice in a T2A synthesis API (the preview/audition within this API does not count).
* Voices produced via this rapid cloning API are **temporary**. To keep a cloned voice permanently, call **any** T2A speech synthesis API with that voice **within 168 hours (7 days)** (the preview/audition within this API does not count). If the time limit is exceeded, the voice will be deleted.
* This API is stateless: each call only processes the incoming data, does not store user-uploaded content, and involves no business-logic state.

<Columns cols={2}>
  <Card title="Upload Clone Audio" icon="upload" href="/docs/api-reference/voice-cloning-uploadcloneaudio" cta="View Docs">
    Upload audio file to clone
  </Card>

  <Card title="Clone Voice" icon="mic" href="/docs/api-reference/voice-cloning-clone" cta="View Docs">
    Execute voice cloning
  </Card>
</Columns>

***

## Voice Design

This API supports generating personalized custom voices based on user-provided voice description prompts.

The generated voices (voice\_id) can then be used in the T2A API and the T2A Async API for speech generation.

**Supported Models**

> It is recommended to use **speech-02-hd** for the best results.

| Model            | Description                                                                                              |
| :--------------- | :------------------------------------------------------------------------------------------------------- |
| speech-2.8-hd    | Latest HD model. Ultra-realistic quality featuring sound tags.                                           |
| speech-2.8-turbo | Latest Turbo model. Seamless speed meets natural flow.                                                   |
| speech-2.6-hd    | HD model with real-time response, intelligent parsing, fluent LoRA voice                                 |
| speech-2.6-turbo | Turbo model. Ultimate Value, 40 Languages                                                                |
| speech-02-hd     | Superior rhythm and stability, with outstanding performance in replication similarity and sound quality. |
| speech-02-turbo  | Superior rhythm and stability, with enhanced multilingual capabilities and excellent performance.        |

### Notes

> * Using this API to generate a voice does not immediately incur a fee. The generation fee will be charged upon the first use of the generated voice in speech synthesis.
> * Voices generated through this API are temporary. If you wish to keep a voice permanently, you must use it in any speech synthesis API within 168 hours (7 days).

<Card title="Voice Design API" icon="wand-magic-sparkles" href="/docs/api-reference/voice-design-design" cta="View Docs">
  Generate personalized voices from descriptions
</Card>

***

## Image Generation

This API supports images generations from text or references, allowing custom aspect ratios and resolutions for diverse needs.

**API Description**

You can generate images by creating an image generation task using text prompts and/or reference images.

**Model List**

| Model    | Description                                                                                                                                                              |
| :------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| image-01 | A high-quality image generation model that produces fine-grained details. Supports both text-to-image and image-to-image generation (with subject reference for people). |

<Columns cols={2}>
  <Card title="Text to Image" icon="file-text" href="/docs/api-reference/image-generation-t2i" cta="View Docs">
    Generate image from text description
  </Card>

  <Card title="Image to Image" icon="image-plus" href="/docs/api-reference/image-generation-i2i" cta="View Docs">
    Generate image from reference image
  </Card>
</Columns>

***

## Music Generation

This API generates a vocal song based on a music description (prompt) and lyrics.

**Models**

| Model     | Usage                                                                                                                  |
| :-------- | :--------------------------------------------------------------------------------------------------------------------- |
| music-3.0 | The latest music generation model. Supports user-provided musical inspiration and lyrics to create AI-generated music. |

<Card title="Music Generation API" icon="music" href="/docs/api-reference/music-generation" cta="View Docs">
  Generate music from description and lyrics
</Card>

***

## File Management

This API is for file management and is used with other MiniMax APIs.

**API Description**

This API includes 5 endpoints: **Upload**, **List**, **Retrieve**, **Retrieve Content**, **Delete**.

Supported file formats, capacity, and size limits are defined by the **Upload File** API documentation — see [Upload File](/docs/api-reference/file-management-upload).

<Columns cols={2}>
  <Card title="Upload File" icon="upload" href="/docs/api-reference/file-management-upload" cta="View Docs">
    Upload files to the platform
  </Card>

  <Card title="List Files" icon="list" href="/docs/api-reference/file-management-list" cta="View Docs">
    Get list of uploaded files
  </Card>
</Columns>

***

## Official MCP

MiniMax provides official Model Context Protocol (MCP) server implementations:

* [Python version](https://github.com/MiniMax-AI/MiniMax-MCP)
* [JavaScript version](https://github.com/MiniMax-AI/MiniMax-MCP-JS)

Both support speech synthesis, voice cloning, video generation, and music generation. For details, refer to the [MiniMax MCP User Guide](/docs/guides/mcp-guide).

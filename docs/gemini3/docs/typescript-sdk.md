# Google Gen AI SDK for TypeScript and JavaScript

**Source URL:** [https://googleapis.github.io/js-genai/release_docs/index.html](https://googleapis.github.io/js-genai/release_docs/index.html)

---

## Overview

The Google Gen AI JavaScript SDK is designed for TypeScript and JavaScript developers to build applications powered by Gemini. The SDK supports both the **Gemini Developer API** and **Vertex AI**.

The Google Gen AI SDK is designed to work with Gemini 2.0+ features.

> **Caution: API Key Security**
> Avoid exposing API keys in client-side code. Use server-side implementations in production environments.

---

## Code Generation

Generative models are often unaware of recent API and SDK updates and may suggest outdated or legacy code.

We recommend using our Code Generation instructions `codegen_instructions.md` when generating Google Gen AI SDK code to guide your model towards using the more recent SDK features. Copy and paste the instructions into your development environment to provide the model with the necessary context.

---

## Prerequisites

1. Node.js version 20 or later

### For Vertex AI users (excluding Vertex AI Studio)

1. Select or create a Google Cloud project.
2. Enable billing for your project.
3. Enable the Vertex AI API.
4. Configure authentication for your project:
   - Install the gcloud CLI.
   - Initialize the gcloud CLI.
   - Create local authentication credentials:
     ```bash
     gcloud auth application-default login
     ```

A list of accepted authentication options are listed in the `GoogleAuthOptions` interface of the `google-auth-library-node.js` GitHub repo.

---

## Installation

To install the SDK, run the following command:

```bash
npm install @google/genai
```

---

## Quickstart

The simplest way to get started is to use an API key from **Google AI Studio**:

```javascript
import {GoogleGenAI} from '@google/genai';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});

async function main() {
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: 'Why is the sky blue?',
  });
  console.log(response.text);
}

main();
```

---

## Initialization

The Google Gen AI SDK provides support for both the Google AI Studio and Vertex AI implementations of the Gemini API.

### Gemini Developer API

For server-side applications, initialize using an API key from Google AI Studio:

```javascript
import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({apiKey: 'GEMINI_API_KEY'});
```

#### Browser

> **Caution: API Key Security**
> Avoid exposing API keys in client-side code. Use server-side implementations in production environments.

In the browser, the initialization code is identical:

```javascript
import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({apiKey: 'GEMINI_API_KEY'});
```

### Vertex AI

Sample code for Vertex AI initialization:

```javascript
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({
    vertexai: true,
    project: 'your_project',
    location: 'your_location',
});
```

### (Optional) Using Environment Variables (NodeJS only)

**Gemini Developer API:**
```bash
export GOOGLE_API_KEY='your-api-key'
```

**Gemini API on Vertex AI:**
```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

```javascript
import {GoogleGenAI} from '@google/genai';
const ai = new GoogleGenAI();
```

---

## API Selection

By default, the SDK uses the beta API endpoints. Stable API endpoints can be selected by setting the API version to `v1`.

**Vertex AI:**
```javascript
const ai = new GoogleGenAI({
    vertexai: true,
    project: 'your_project',
    location: 'your_location',
    apiVersion: 'v1'
});
```

**Gemini Developer API:**
```javascript
const ai = new GoogleGenAI({
    apiKey: 'GEMINI_API_KEY',
    apiVersion: 'v1alpha'
});
```

---

## GoogleGenAI Overview

All API features are accessed through an instance of the `GoogleGenAI` class. Submodules include:

- **`ai.models`**: Query models (`generateContent`, `generateImages`, etc.) or examine metadata.
- **`ai.caches`**: Create and manage caches to reduce costs for large prompt prefixes.
- **`ai.chats`**: Create local stateful chat objects for multi-turn interactions.
- **`ai.files`**: Upload files to the API and reference them in prompts.
- **`ai.live`**: Start a live session for real-time interaction (text, audio, video).

---

## Samples

### Streaming

```javascript
import {GoogleGenAI} from '@google/genai';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});

async function main() {
  const response = await ai.models.generateContentStream({
    model: 'gemini-2.5-flash',
    contents: 'Write a 100-word poem.',
  });
  for await (const chunk of response) {
    console.log(chunk.text);
  }
}

main();
```

### Function Calling

1. Declare the function name, description, and schema.
2. Call `generateContent` with function calling enabled.
3. Use returned `FunctionCall` parameters to call your function.
4. Send the result back as a `FunctionResponse`.

```javascript
import {GoogleGenAI, FunctionCallingConfigMode, FunctionDeclaration, Type} from '@google/genai';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

async function main() {
  const controlLightDeclaration: FunctionDeclaration = {
    name: 'controlLight',
    parametersJsonSchema: {
      type: 'object',
      properties:{
        brightness: { type:'number' },
        colorTemperature: { type:'string' },
      },
      required: ['brightness', 'colorTemperature'],
    },
  };

  const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: 'Dim the lights so the room feels cozy and warm.',
    config: {
      toolConfig: {
        functionCallingConfig: {
          mode: FunctionCallingConfigMode.ANY,
          allowedFunctionNames: ['controlLight'],
        }
      },
      tools: [{functionDeclarations: [controlLightDeclaration]}]
    }
  });

  console.log(response.functionCalls);
}

main();
```

### Model Context Protocol (MCP) Support (Experimental)

```javascript
import { GoogleGenAI, FunctionCallingConfigMode , mcpToTool} from '@google/genai';
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const serverParams = new StdioClientTransport({
  command: "npx",
  args: ["-y", "@philschmid/weather-mcp"]
});

const client = new Client({ name: "example-client", version: "1.0.0" });
const ai = new GoogleGenAI({});

await client.connect(serverParams);

const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: `What is the weather in London?`,
  config: {
    tools: [mcpToTool(client)],
  },
});
console.log(response.text);
await client.close();
```

---

## Generate Content Structure

The `contents` parameter accepts:
- **Content**: Singular instance (wrapped in an array).
- **Content[]**: No transformation.
- **Part | string**: Wrapped in a `Content` instance with role 'user'.
- **Part[] | string[]**: Wrapped into a single `Content` with role 'user'.

*Note: Does not apply to `FunctionCall` and `FunctionResponse` parts.*

---

## Error Handling

```javascript
import {GoogleGenAI} from '@google/genai';
const ai = new GoogleGenAI({apiKey: process.env.GEMINI_API_KEY});

async function main() {
  await ai.models.generateContent({
    model: 'non-existent-model',
    contents: 'Hello',
  }).catch((e) => {
    console.error('error name: ', e.name);
    console.error('error message: ', e.message);
    console.error('error status: ', e.status);
  });
}

main();
```

---

## Multimodal Output

```javascript
import * as fs from 'fs';

const interaction = await ai.interactions.create({
  model: 'gemini-3-pro-image-preview',
  input: 'Generate an image of a futuristic city.',
  response_modalities: ['image'],
});

for (const output of interaction.outputs!) {
  if (output.type === 'image') {
    console.log(`Generated image with mime_type: ${output.mime_type}`);
    fs.writeFileSync('generated_city.png', Buffer.from(output.data!, 'base64'));
  }
}
```

---

## Comparison with Other SDKs

- **@google/genai**: Google Deepmind's "vanilla" SDK for generative AI, supporting both Vertex AI and Gemini Developer platforms.
- **@google/generative_language** & **@google-cloud/vertexai**: Previous iterations, no longer receiving Gemini 2.0+ features.

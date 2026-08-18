> ## Documentation Index
> Fetch the complete documentation index at: https://docs.z.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Tool Integration

## Supported Tools

<Tip>
  The GLM Coding Plan is limited to use within the following officially supported tools and product environments; users may not use their subscription benefits for tools or scenarios outside of this scope.
</Tip>

### 1. Coding Agent Tool

Click on the tool documentation below that you wish to use, and follow the corresponding integration guide to set it up.

<CardGroup cols={3}>
  <Card title="ZCode" href="/devpack/tool/zcode" icon="hammer">
    ZCode integrates AI agents into your existing toolchain.
  </Card>

  <Card title="Claude Code" href="/devpack/tool/claude" icon="puzzle-piece">
    The Claude Code IDE plugin supports VSCode and JetBrains.
  </Card>

  <Card title="Claude for IDE" href="/devpack/tool/claude-for-ide" icon="stars">
    The Claude Code IDE plugin supports VSCode and JetBrains.
  </Card>

  <Card title="Codex" href="/devpack/tool/codex" icon="hammer">
    OpenAI's AI coding agent that helps you write, review, and debug code.
  </Card>

  <Card title="OpenCode" href="/devpack/tool/opencode" icon="globe">
    The Claude Code IDE plugin supports VSCode and JetBrains.
  </Card>

  <Card title="Pi" href="/devpack/tool/pi" icon="list">
    A minimalist terminal coding agent with a small core, extensible via extensions, Skills, and Pi Packages.
  </Card>

  <Card title="Cursor" href="/devpack/tool/cursor" icon="code">
    An AI-first code editor that supports custom model configurations.
  </Card>

  <Card title="Cline" href="/devpack/tool/cline" icon="brain">
    An AI programming extension for VS Code that supports code generation and file operations.
  </Card>

  <Card title="TRAE" href="/devpack/tool/trae" icon="headset">
    An AI editor capable of independently completing various development tasks.
  </Card>

  <Card title="Qoder" href="/devpack/tool/qoder" icon="book">
    An agentic coding platform designed for real software development.
  </Card>

  <Card title="Droid" href="/devpack/tool/droid" icon="helmet-safety">
    Enterprise-grade AI coding agent that runs in the terminal to handle end-to-end workflows.
  </Card>

  <Card title="Kilo Code" href="/devpack/tool/kilo" icon="bolt">
    A powerful VS Code extension for code generation and project management.
  </Card>

  <Card title="Roo Code" href="/devpack/tool/roo" icon="box">
    A smart VS Code extension for code writing and refactoring.
  </Card>

  <Card title="Crush" href="/devpack/tool/crush" icon="gauge-high">
    A terminal-based AI programming tool that supports both CLI and TUI interfaces.
  </Card>

  <Card title="Goose" href="/devpack/tool/goose" icon="rocket">
    AI Agent tool, supporting local execution and automated engineering tasks.
  </Card>

  <Card title="Eigent" href="/devpack/tool/eigent" icon="sparkles">
    A desktop AI agent built on a multi-agent architecture, capable of automating browser, terminal, and MCP-powered workflows.
  </Card>
</CardGroup>

### 2. General-purpose Agent Tool

<Tip>
  The general-purpose agent tools listed below are also supported and will continue to be served on a best-effort basis. Under high inference load, some requests may face temporary rate limits.
</Tip>

<CardGroup cols={2}>
  <Card title="OpenClaw" href="/devpack/tool/openclaw" icon="lobster">
    An open-source AI assistant that runs on local devices, supports multi-platform use.
  </Card>

  <Card title="Hermes Agent" href="#" icon="shield">
    An open-source evolving AI agent with persistent memory,getting smarter with use.
  </Card>

  <Card title="SillyTavern" href="#" icon="glasses">
    A highly customizable AI chat frontend for immersive roleplay with multi-model & media support.
  </Card>
</CardGroup>

## Coding Endpoint

GLM Coding Plan supports both the Anthropic and OpenAI protocols. Make sure to configure the correct `Base URL`:

| Protocol                | Base URL                              |
| ----------------------- | ------------------------------------- |
| Anthropic Messages      | `https://api.z.ai/api/anthropic`      |
| OpenAI Chat Completions | `https://api.z.ai/api/coding/paas/v4` |
| OpenAI Responses        | `https://api.z.ai/api/v1`             |

**Core Steps**

1. Choose the protocol that fits your tool (Anthropic Messages or OpenAI Chat Completions).
2. Configure the correct Base URL.
3. Enter your API Key and select a GLM model.

<Warning>
  Please select the correct endpoint address based on the tool you are using. Incorrect endpoint configuration will result in inability to use GLM Coding Plan subscription quota.
</Warning>

## Config Example

Using **Cline** as an example, the following steps demonstrate how to integrate GLM models via the OpenAI Compatible protocol. Similarly, other tools supporting the OpenAI-compatible protocol can adopt the same configuration approach.

### 1. Install the Cline Plugin

1. Open VS Code and click the Extensions Marketplace icon on the left.
2. Enter `cline` in the search box and locate the `Cline` extension.
3. Click `Install` to install it, then choose to trust the developer.

### 2. Configure API Endpoint

In Cline, select `Use your own API Key`, then fill in the following configuration:

* **API Provider**: Select `OpenAI Compatible`
* **Base URL**: Enter `https://api.z.ai/api/coding/paas/v4`
* **API Key**: Enter your Z.AI API Key
* **Model**: Select "Use custom" and enter the GLM model code you want to use (e.g., `glm-5.2`)
* **Other Configurations**:
  * Uncheck **Support Images**
  * Adjust **Context Window Size** based on your model (`glm-5.2` is `1000000`; other models `200000`)

### 3. Get Started

Once configured, you can enter your requirements in the input box to let the model assist you with code generation, file editing, refactoring, explaining code logic, debugging, and more.

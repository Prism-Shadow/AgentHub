> Fetch the complete documentation index at: https://platform.kimi.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# 使用 Kimi API 的流式输出功能

Kimi 大模型收到问题后会先进行推理，再逐个 Token 生成回答；流式输出（Streaming）让模型每生成一定数量的 Tokens（通常是 1 个 Token）就立即发送给客户端，而不是等全部生成完毕再一次性返回。等待完整回复通常要数秒，问题复杂、回复较长时可能拉长到 10 秒甚至 20 秒；开启流式输出后，用户能第一时间看到第一个 Token，显著减少等待时间。当你与 [Kimi 智能助手](https://kimi.com) 对话时，回复逐字“跳”出来，就是流式输出的效果。

## 开启流式输出

在请求中设置 `stream=True` 即可开启流式输出。此时 SDK 返回一个可迭代对象，用循环逐个读取数据块（chunk）：每个 chunk 的结构与 completion 相似，但 `message` 字段被替换为 `delta` 字段。

<Note>
  本页示例默认使用最新模型 `kimi-k3`。K3 使用请求顶层 `reasoning_effort` 配置思考力度（支持 `"low"` / `"high"` / `"max"`，默认 `"max"`）。换用 `kimi-k2.6`、`kimi-k2.5` 等其他模型时，只需替换 `model` 字段，但各模型的参数配置存在差异，详见[模型参数参考](/docs/api/models-overview)。
</Note>

<Tabs>
  <Tab title="python">
    ```python theme={null}
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key = os.environ["MOONSHOT_API_KEY"], # 运行前请设置 MOONSHOT_API_KEY 环境变量
        base_url = "https://api.moonshot.cn/v1",
    )

    stream = client.chat.completions.create(
        model = "kimi-k3",
        messages = [
            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
            {"role": "user", "content": "你好，我叫李雷，1+1等于多少？"}
        ],
        stream=True, # <-- 注意这里，我们通过设置 stream=True 开启流式输出模式
    )

    # 当启用流式输出模式（stream=True），SDK 返回的内容也发生了变化，我们不再直接访问返回值中的 choice
    # 而是通过 for 循环逐个访问返回值中每个单独的块（chunk）

    for chunk in stream:
    	# 在这里，每个 chunk 的结构都与之前的 completion 相似，但 message 字段被替换成了 delta 字段
    	delta = chunk.choices[0].delta # <-- message 字段被替换成了 delta 字段

    	if delta.content:
    		# 我们在打印内容时，由于是流式输出，为了保证句子的连贯性，我们不人为地添加
    		# 换行符，因此通过设置 end="" 来取消 print 自带的换行符。
    		print(delta.content, end="")
    ```
  </Tab>

  <Tab title="node.js">
    ```js theme={null}
    const OpenAI = require('openai')

    const client = new OpenAI({
        apiKey: process.env.MOONSHOT_API_KEY, // 运行前请设置 MOONSHOT_API_KEY 环境变量
        baseURL: "https://api.moonshot.cn/v1",
    })

    async function main() {
        const stream = await client.chat.completions.create({
            model: "kimi-k3",
            messages: [
                {role: "system", content: "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
                {role: "user", content: "你好，我叫李雷，1+1等于多少？"}
            ],
            stream: true, // <-- 注意这里，我们通过设置 stream=True 开启流式输出模式
        })

        // 当启用流式输出模式（stream=True），SDK 返回的内容也发生了变化，我们不再直接访问返回值中的 choice
        // 而是通过 for 循环逐个访问返回值中每个单独的块（chunk）

        for await (chunk of stream) {
            // 在这里，每个 chunk 的结构都与之前的 completion 相似，但 message 字段被替换成了 delta 字段
            delta = chunk.choices[0].delta // <-- message 字段被替换成了 delta 字段

            if (delta.content) {
                // 我们在打印内容时，由于是流式输出，为了保证句子的连贯性，我们不人为地添加
                // 换行符，因此通过设置 end="" 来取消 print 自带的换行符。
                console.log(delta.content, end="")
            }
        }
    }

    main()
    ```
  </Tab>
</Tabs>

## 解析 SSE 响应体

开启流式输出后，接口不再返回 JSON 格式的响应（`Content-Type: application/json`），而是返回 `Content-Type: text/event-stream`（SSE），服务端得以源源不断地向客户端传输 Tokens。[SSE](https://kimi.com/share/cr7boh3dqn37a5q9tds0) 的响应体如下所示：

```text theme={null}
data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k3","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k3","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}

...

data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k3","choices":[{"index":0,"delta":{"content":"。"},"finish_reason":null}]}

data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k3","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":19,"completion_tokens":13,"total_tokens":32}}]}

data: [DONE]
```

响应体中的每个数据块均以 `data: ` 为前缀，紧跟一个合法的 JSON 对象，并以两个换行符 `\n\n` 结束。所有数据块传输完成后，服务端发送 `data: [DONE]` 标识传输结束，此时可断开网络连接。

*注意：请始终使用 `data: [DONE]` 判断数据是否传输完成，而不是使用 `finish_reason` 或其他方式。如果未收到 `data: [DONE]`，即使已经获取了 `finish_reason=stop`，也不应视作传输完成；换句话说，在收到 `data: [DONE]` 之前，都应视作 **消息是不完整的**。*

流式输出过程中会有 `content` 字段会逐块下发；`role` 和 `usage` 不会在每个数据块中重复出现——`role` 仅出现在第一个数据块，`usage` 仅出现在最后一个数据块。

## 统计 Tokens 用量

计算 Tokens 有两种方式。最直接、最准确的一种，是等所有数据块传输完毕后，读取最后一个数据块中的 `usage` 字段，查看本次请求产生的 `prompt_tokens`/`completion_tokens`/`total_tokens`：

```text theme={null}
...

data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k3","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":19,"completion_tokens":13,"total_tokens":32}}]}
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                               通过访问最后一个数据块中的 usage 字段来查看当前请求产生的 Tokens 数量
data: [DONE]
```

<Note>
  注意 `usage` 嵌套在最后一个数据块的 `choices[0]` 内（即 `choices[0].usage`），而非数据块顶层。使用 OpenAI SDK 时 `chunk.usage` 为 `None`，请读取 `chunk.choices[0].usage`，或自行解析原始 SSE 数据块。
</Note>

但流式输出可能因网络连接中断、客户端程序错误等不可控因素被打断，此时最后一个数据块尚未到达，也就无从得知本次请求消耗的 Tokens。为避免统计失败，建议保存已收到的每个数据块的内容，并在请求结束后（无论是否成功结束）调用 Tokens 计算接口统计实际消耗量：

<Tabs>
  <Tab title="python">
    ```python theme={null}
    import os
    import httpx
    from openai import OpenAI

    client = OpenAI(
        api_key = os.environ["MOONSHOT_API_KEY"], # 运行前请设置 MOONSHOT_API_KEY 环境变量
        base_url = "https://api.moonshot.cn/v1",
    )

    stream = client.chat.completions.create(
        model = "kimi-k3",
        messages = [
            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
            {"role": "user", "content": "你好，我叫李雷，1+1等于多少？"}
        ],
        stream=True, # <-- 注意这里，我们通过设置 stream=True 开启流式输出模式
    )


    def estimate_token_count(input: str) -> int:
        """
        在这里实现你的 Tokens 计算逻辑，或是直接调用我们的 Tokens 计算接口计算 Tokens

        https://api.moonshot.cn/v1/tokenizers/estimate-token-count
        """
        header = {
            "Authorization": f"Bearer {os.environ['MOONSHOT_API_KEY']}",
        }
        data = {
            "model": "kimi-k3",
            "messages": [
                {"role": "user", "content": input},
            ]
        }
        r = httpx.post("https://api.moonshot.cn/v1/tokenizers/estimate-token-count", headers=header, json=data)
        r.raise_for_status()
        return r.json()["data"]["total_tokens"]


    completion = []
    for chunk in stream:
    	delta = chunk.choices[0].delta
    	if delta.content:
    		completion.append(delta.content)


    print("completion_tokens:", estimate_token_count("".join(completion)))
    ```
  </Tab>

  <Tab title="node.js">
    ```js theme={null}
    const axios = require('axios');
    const OpenAI = require('openai');

    client = new OpenAI({
        apiKey: process.env.MOONSHOT_API_KEY,
        baseURL: "https://api.moonshot.cn/v1",
    })


    async function estimate_token_count(input_messages) {
        /*
        在这里实现你的 Tokens 计算逻辑，或是直接调用我们的 Tokens 计算接口计算 Tokens

        https://api.moonshot.cn/v1/tokenizers/estimate-token-count
        */
        header = {
            "Authorization": `Bearer ${process.env.MOONSHOT_API_KEY}`,
        }
        data = {
            "model": "kimi-k3",
            "messages": input_messages,
        }
        r = await axios.post("https://api.moonshot.cn/v1/tokenizers/estimate-token-count", data, {headers: header})
        .catch(function (error) {
            console.log(error)
        })
        return r.data.data.total_tokens
    }

    async function main() {

        const stream = await client.chat.completions.create({
            model: "kimi-k3",
            messages: [
                {role: "system", content: "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
                {role: "user", content: "你好，我叫李雷，1+1等于多少？"}
            ],
            stream: true, // <-- 注意这里，我们通过设置 stream=True 开启流式输出模式
        })

        const completion = [];
        for await (chunk of stream) {
            const delta = chunk.choices[0].delta
            if (delta.content) {
                completion.push(delta.content)
            }
        }

        console.log("completion_tokens:", await estimate_token_count(completion.join("")))
    }

    main()
    ```
  </Tab>
</Tabs>

## 终止流式输出

需要提前终止输出时，直接关闭 HTTP 网络连接或丢弃后续数据块即可，例如在循环中 `break`：

```python theme={null}
for chunk in stream:
	if condition:
		break
```

## 不用 SDK 直接处理 SSE

在没有 SDK 的语言环境，或 SDK 无法满足你的业务逻辑时，可以直接对接 HTTP 接口来处理流式输出。以下示例演示如何逐行读取并解析 [SSE](https://kimi.com/share/cr7boh3dqn37a5q9tds0) 响应体，详细说明见代码注释：

<Tabs>
  <Tab title="python">
    ```python theme={null}
    import os
    import json
    import httpx # 我们使用 httpx 库来执行我们的 HTTP 请求


    data = {
    	"model": "kimi-k3",
    	"messages": [
    		# 具体的 messages
    	],
    	"stream": True,
    }


    # 使用 httpx 向 Kimi 大模型发出 chat 请求，并获得响应 r
    r = httpx.post("https://api.moonshot.cn/v1/chat/completions", headers={"Authorization": f"Bearer {os.environ['MOONSHOT_API_KEY']}"}, json=data)
    if r.status_code != 200:
    	raise Exception(r.text)


    data: str

    # 在这里，我们使用了 iter_lines 方法来逐行读取响应体
    for line in r.iter_lines():
    	# 去除每一行收尾的空格，以便更好地处理数据块
    	line = line.strip()

    	# 接下来我们要处理三种不同的情况：
    	#   1. 如果当前行是空行，则表明前一个数据块已接收完毕（即前文提到的，通过两个换行符结束数据块传输），我们可以对该数据块进行反序列化，并打印出对应的 content 内容；
    	#   2. 如果当前行为非空行，且以 data: 开头，则表明这是一个数据块传输的开始，我们去除 data: 前缀后，首先判断是否是结束符 [DONE]，如果不是，将数据内容保存到 data 变量；
    	#   3. 如果当前行为非空行，但不以 data: 开头，则表明当前行仍然归属上一个正在传输的数据块，我们将当前行的内容追加到 data 变量尾部；

    	if len(line) == 0:
    		chunk = json.loads(data)

    		# 这里的处理逻辑可以替换成你的业务逻辑，打印仅是为了展示处理流程
    		choice = chunk["choices"][0]
    		usage = choice.get("usage")
    		if usage:
    			print("total_tokens:", usage["total_tokens"])
    		delta = choice["delta"]
    		role = delta.get("role")
    		if role:
    			print("role:", role)
    		content = delta.get("content")
    		if content:
    			print(content, end="")

    		data = "" # 重置 data
    	elif line.startswith("data: "):
    		data = line.lstrip("data: ")

    		# 当数据块内容为 [DONE] 时，则表明所有数据块已发送完毕，可断开网络连接
    		if data == "[DONE]":
    			break
    	else:
    		data = data + "\n" + line # 我们仍然在追加内容时，为其添加一个换行符，因为这可能是该数据块有意将数据分行展示
    ```
  </Tab>

  <Tab title="node.js">
    ```js theme={null}
    const axios = require('axios'); // 使用 axios 库来执行 HTTP 请求

    let data = {
        "model": "kimi-k3",
        "messages": [
            // 具体的 messages
        ],
        "stream": true,
    };

    // 使用 axios 向 Kimi 大模型发出 chat 请求，并获得响应 r
    axios.post("https://api.moonshot.cn/v1/chat/completions", data, {
        responseType: 'stream'
    }).then(response => {
        let data = '';
        response.data.on('data', chunk => {
            // 去除每一行收尾的空格，以便更好地处理数据块
            let line = chunk.toString().trim();

            if (line === '') {
                try {
                    let chunk = JSON.parse(data);
                    let choice = chunk.choices[0];
                    let usage = choice.usage;
                    if (usage) {
                        console.log("total_tokens:", usage.total_tokens);
                    }
                    let delta = choice.delta;
                    let role = delta.role;
                    if (role) {
                        console.log("role:", role);
                    }
                    let content = delta.content;
                    if (content) {
                        console.log(content);
                    }
                } catch (error) {
                    console.error("Error parsing JSON:", error);
                }
                data = ''; // 重置 data
            } else if (line.startsWith('data: ')) {
                data = line.substring(6);
                if (data === '[DONE]') {
                    response.data.destroy();
                }
            } else {
                data += '\n' + line;
            }
        });
    }).catch(error => {
        console.error("Error in request:", error);
    });
    ```
  </Tab>
</Tabs>

无论使用哪种语言，处理流式输出的基本步骤相同：

1. 发起 HTTP 请求，并在请求体中将 `stream` 参数设置为 `true`；
2. 检查响应 `Headers` 中的 `Content-Type`，为 `text/event-stream` 即表示当前响应是流式输出；
3. 逐行读取响应内容并解析数据块（JSON 格式），通过 `data: ` 前缀和换行符 `\n` 判断数据块的起止位置；
4. 数据块内容为 `[DONE]` 时表示传输完成。

## 多个回复（`n` 参数）

<Note>
  当前模型（`kimi-k3`、`kimi-k2.7-code`、`kimi-k2.6`）的 `n` 固定为 `1`，暂不支持一次请求返回多个回复；传入大于 1 的 `n` 会返回 400 错误（`invalid n: only 1 is allowed for this model`），流式与非流式请求均如此。各模型的参数约束详见[模型参数参考](/docs/api/models-overview)。
</Note>

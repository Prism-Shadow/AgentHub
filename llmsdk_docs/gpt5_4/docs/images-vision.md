# Images and vision

## Overview

<div className="mb-10 w-full max-w-full overflow-hidden">
  </div>

In this guide, you will learn about building applications involving images with the OpenAI API.
If you know what you want to build, find your use case below to get started. If you're not sure where to start, continue reading to get an overview.

### A tour of image-related use cases

Recent language models can process image inputs and analyze them — a capability known as **vision**. With `gpt-image-1`, they can both analyze visual inputs and create images.

The OpenAI API offers several endpoints to process images as input or generate them as output, enabling you to build powerful multimodal applications.

| API                                                  | Supported use cases                                                   |
| ---------------------------------------------------- | --------------------------------------------------------------------- |
| [Responses API](https://developers.openai.com/api/docs/api-reference/responses)   | Analyze images and use them as input and/or generate images as output |
| [Images API](https://developers.openai.com/api/docs/api-reference/images)         | Generate images as output, optionally using images as input           |
| [Chat Completions API](https://developers.openai.com/api/docs/api-reference/chat) | Analyze images and use them as input to generate text or audio        |

To learn more about the input and output modalities supported by our models, refer to our [models page](https://developers.openai.com/api/docs/models).

## Generate or edit images

You can generate or edit images using the Image API or the Responses API.

Our latest image generation model, `gpt-image-1`, is a natively multimodal large language model.
It can understand text and images and leverage its broad world knowledge to generate images with better instruction following and contextual awareness.

In contrast, we also offer specialized image generation models - DALL·E 2 and 3 - which don't have the same inherent understanding of the world as GPT Image.



Generate images with Responses

```javascript
import OpenAI from "openai";
const openai = new OpenAI();

const response = await openai.responses.create({
    model: "gpt-4.1-mini",
    input: "Generate an image of gray tabby cat hugging an otter with an orange scarf",
    tools: [{type: "image_generation"}],
});

// Save the image to a file
const imageData = response.output
  .filter((output) => output.type === "image_generation_call")
  .map((output) => output.result);

if (imageData.length > 0) {
  const imageBase64 = imageData[0];
  const fs = await import("fs");
  fs.writeFileSync("cat_and_otter.png", Buffer.from(imageBase64, "base64"));
}
```

```python
from openai import OpenAI
import base64

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
    tools=[{"type": "image_generation"}],
)

// Save the image to a file
image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]

if image_data:
    image_base64 = image_data[0]
    with open("cat_and_otter.png", "wb") as f:
        f.write(base64.b64decode(image_base64))
```



You can learn more about image generation in our [Image
  generation](https://developers.openai.com/api/docs/guides/image-generation) guide.

### Using world knowledge for image generation

The difference between DALL·E models and GPT Image is that a natively multimodal language model can use its visual understanding of the world to generate lifelike images including real-life details without a reference.

For example, if you prompt GPT Image to generate an image of a glass cabinet with the most popular semi-precious stones, the model knows enough to select gemstones like amethyst, rose quartz, jade, etc, and depict them in a realistic way.

## Analyze images

**Vision** is the ability for a model to "see" and understand images. If there is text in an image, the model can also understand the text.
It can understand most visual elements, including objects, shapes, colors, and textures, even if there are some [limitations](#limitations).

### Giving a model images as input





You can provide images as input to generation requests in multiple ways:

- By providing a fully qualified URL to an image file
- By providing an image as a Base64-encoded data URL
- By providing a file ID (created with the [Files API](https://developers.openai.com/api/docs/api-reference/files))

You can provide multiple images as input in a single request by including multiple images in the `content` array, but keep in mind that [images count as tokens](#calculating-costs) and will be billed accordingly.



<div data-content-switcher-pane data-value="url">
    <div class="hidden">Passing a URL</div>
    Analyze the content of an image

```javascript
import OpenAI from "openai";

const openai = new OpenAI();

const response = await openai.responses.create({
    model: "gpt-4.1-mini",
    input: [{
        role: "user",
        content: [
            { type: "input_text", text: "what's in this image?" },
            {
                type: "input_image",
                image_url: "https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg",
            },
        ],
    }],
});

console.log(response.output_text);
```

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {
                "type": "input_image",
                "image_url": "https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg",
            },
        ],
    }],
)

print(response.output_text)
```

```csharp
using OpenAI.Responses;

string key = Environment.GetEnvironmentVariable("OPENAI_API_KEY")!;
OpenAIResponseClient client = new(model: "gpt-5", apiKey: key);

Uri imageUrl = new("https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg");

OpenAIResponse response = (OpenAIResponse)client.CreateResponse([
    ResponseItem.CreateUserMessageItem([
        ResponseContentPart.CreateInputTextPart("What is in this image?"),
        ResponseContentPart.CreateInputImagePart(imageUrl)
    ])
]);

Console.WriteLine(response.GetOutputText());
```

```bash
curl https://api.openai.com/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -d '{
    "model": "gpt-4.1-mini",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "what is in this image?"},
          {
            "type": "input_image",
            "image_url": "https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg"
          }
        ]
      }
    ]
  }'
```

  </div>
  <div data-content-switcher-pane data-value="base64-encoded" hidden>
    <div class="hidden">Passing a Base64 encoded image</div>
    Analyze the content of an image

```javascript
import fs from "fs";
import OpenAI from "openai";

const openai = new OpenAI();

const imagePath = "path_to_your_image.jpg";
const base64Image = fs.readFileSync(imagePath, "base64");

const response = await openai.responses.create({
    model: "gpt-4.1-mini",
    input: [
        {
            role: "user",
            content: [
                { type: "input_text", text: "what's in this image?" },
                {
                    type: "input_image",
                    image_url: \`data:image/jpeg;base64,\${base64Image}\`,
                },
            ],
        },
    ],
});

console.log(response.output_text);
```

```python
import base64
from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)


response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": "what's in this image?" },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)

print(response.output_text)
```

```csharp
using OpenAI.Responses;

string key = Environment.GetEnvironmentVariable("OPENAI_API_KEY")!;
OpenAIResponseClient client = new(model: "gpt-5", apiKey: key);

Uri imageUrl = new("https://openai-documentation.vercel.app/images/cat_and_otter.png");
using HttpClient http = new();

// Download an image as stream
using var stream = await http.GetStreamAsync(imageUrl);

OpenAIResponse response1 = (OpenAIResponse)client.CreateResponse([
    ResponseItem.CreateUserMessageItem([
        ResponseContentPart.CreateInputTextPart("What is in this image?"),
        ResponseContentPart.CreateInputImagePart(BinaryData.FromStream(stream), "image/png")
    ])
]);

Console.WriteLine($"From image stream: {response1.GetOutputText()}");

// Download an image as byte array
byte[] bytes = await http.GetByteArrayAsync(imageUrl);

OpenAIResponse response2 = (OpenAIResponse)client.CreateResponse([
    ResponseItem.CreateUserMessageItem([
        ResponseContentPart.CreateInputTextPart("What is in this image?"),
        ResponseContentPart.CreateInputImagePart(BinaryData.FromBytes(bytes), "image/png")
    ])
]);

Console.WriteLine($"From byte array: {response2.GetOutputText()}");
```

  </div>
  <div data-content-switcher-pane data-value="file" hidden>
    <div class="hidden">Passing a file ID</div>
    Analyze the content of an image

```javascript
import OpenAI from "openai";
import fs from "fs";

const openai = new OpenAI();

// Function to create a file with the Files API
async function createFile(filePath) {
  const fileContent = fs.createReadStream(filePath);
  const result = await openai.files.create({
    file: fileContent,
    purpose: "vision",
  });
  return result.id;
}

// Getting the file ID
const fileId = await createFile("path_to_your_image.jpg");

const response = await openai.responses.create({
  model: "gpt-4.1-mini",
  input: [
    {
      role: "user",
      content: [
        { type: "input_text", text: "what's in this image?" },
        {
          type: "input_image",
          file_id: fileId,
        },
      ],
    },
  ],
});

console.log(response.output_text);
```

```python
from openai import OpenAI

client = OpenAI()

# Function to create a file with the Files API
def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# Getting the file ID
file_id = create_file("path_to_your_image.jpg")

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {
                "type": "input_image",
                "file_id": file_id,
            },
        ],
    }],
)

print(response.output_text)
```

```csharp
using OpenAI.Files;
using OpenAI.Responses;

string key = Environment.GetEnvironmentVariable("OPENAI_API_KEY")!;
OpenAIResponseClient client = new(model: "gpt-5", apiKey: key);

string filename = "cat_and_otter.png";
Uri imageUrl = new($"https://openai-documentation.vercel.app/images/{filename}");
using var http = new HttpClient();

// Download an image as stream
using var stream = await http.GetStreamAsync(imageUrl);

OpenAIFileClient files = new(key);
OpenAIFile file = await files.UploadFileAsync(BinaryData.FromStream(stream), filename, FileUploadPurpose.Vision);

OpenAIResponse response = (OpenAIResponse)client.CreateResponse([
    ResponseItem.CreateUserMessageItem([
        ResponseContentPart.CreateInputTextPart("what's in this image?"),
        ResponseContentPart.CreateInputImagePart(file.Id)
    ])
]);

Console.WriteLine(response.GetOutputText());
```

  </div>




### Image input requirements

Input images must meet the following requirements to be used in the API.

<table>
  <tr>
    <td>Supported file types</td>
    <td>
      - PNG (`.png`) - JPEG (`.jpeg` and `.jpg`) - WEBP (`.webp`) - Non-animated
      GIF (`.gif`)
    </td>
  </tr>
  <tr>
    <td>Size limits</td>
    <td>
      - Up to 50 MB total payload size per request - Up to 500 individual image
      inputs per request
    </td>
  </tr>
  <tr>
    <td>Other requirements</td>
    <td>
      - No watermarks or logos - No NSFW content - Clear enough for a human to
      understand
    </td>
  </tr>
</table>

### Choose an image detail level

The `detail` parameter tells the model what level of detail to use when processing and understanding the image (`low`, `high`, `original`, or `auto` to let the model decide). If you skip the parameter, the model will use `auto`. This behavior is the same in both the Responses API and the Chat Completions API.




  Use the following guidance to choose a detail level:

| Detail level | Best for                                                                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `"low"`      | Fast, low-cost understanding when fine visual detail is not important. The model receives a low-resolution 512px x 512px version of the image. |
| `"high"`     | Standard high-fidelity image understanding.                                                                                                    |
| `"original"` | Large, dense, spatially sensitive, or computer-use images. Available on `gpt-5.4` and future models.                                           |
| `"auto"`     | Let the model choose the detail level.                                                                                                         |

For computer use, localization, and click-accuracy use cases on `gpt-5.4` and future models, we recommend `"detail": "original"`. See the [Computer use guide](https://developers.openai.com/api/docs/guides/tools-computer-use) for more detail.

Read more about how models resize images in the [Model sizing
  behavior](#model-sizing-behavior) section, and about token costs in the
  [Calculating costs](#calculating-costs) section below.

### Model sizing behavior

Different models use different resizing rules before image tokenization:

<table>
  <tr>
    <th>Model family</th>
    <th>Supported detail levels</th>
    <th>Patch and resizing behavior</th>
  </tr>
  <tr>
    <td>
      <code>gpt-5.4</code> and future models
    </td>
    <td>
      <code>low</code>, <code>high</code>, <code>original</code>,
      <code>auto</code>
    </td>
    <td>
      <code>high</code> allows up to 2,500 patches or a 2048-pixel maximum
      dimension. <code>original</code> allows up to 10,000 patches or a
      6000-pixel maximum dimension. If either limit is exceeded, we resize the
      image while preserving aspect ratio to fit within the lesser of those two
      constraints for the selected detail level. [Full resizing details
      below.](#patch-based-image-tokenization)
    </td>
  </tr>
  <tr>
    <td>
      <code>gpt-5-mini</code>, <code>gpt-5-nano</code>, <code>gpt-5.2</code>,
      <code>gpt-5.3-codex</code>, <code>gpt-5-codex-mini</code>,
      <code>gpt-5.1-codex-mini</code>, <code>gpt-5.2-codex</code>,
      <code>gpt-5.2-chat-latest</code>, <code>o4-mini</code>, and the{" "}
      <code>gpt-4.1-mini</code> and <code>gpt-4.1-nano</code> 2025-04-14
      snapshot variants
    </td>
    <td>
      <code>low</code>, <code>high</code>, <code>auto</code>
    </td>
    <td>
      <code>high</code> allows up to 1,536 patches or a 2048-pixel maximum
      dimension. If either limit is exceeded, we resize the image while
      preserving aspect ratio to fit within the lesser of those two constraints.
      [Full resizing details below.](#patch-based-image-tokenization)
    </td>
  </tr>
  <tr>
    <td>
      <code>GPT-4o</code>, <code>GPT-4.1</code>, <code>GPT-4o-mini</code>,
      <code>computer-use-preview</code>, and o-series models except
      <code>o4-mini</code>
    </td>
    <td>
      <code>low</code>, <code>high</code>, <code>auto</code>
    </td>
    <td>
      Use tile-based resizing behavior. See{" "}
      <a href="#gpt-4o-gpt-41-gpt-4o-mini-cua-and-o-series-except-o4-mini">
        the detailed behavior below
      </a>
    </td>
  </tr>
</table>

## Calculating costs

Image inputs are metered and charged in token units similar to text inputs. How images are converted to text token inputs varies based on the model. You can find a vision pricing calculator in the FAQ section of the [pricing page](https://openai.com/api/pricing/).

### Patch-based image tokenization

Some models tokenize images by covering them with 32px x 32px patches. Each model defines a maximum patch budget. The token cost of an image is determined as follows:

A. Compute how many 32px x 32px patches are needed to cover the original image. A patch may extend beyond the image boundary.

```
original_patch_count = ceil(width/32)×ceil(height/32)
```

B. If the original image would exceed the model's patch budget, scale it down proportionally until it fits within that budget. Then adjust the scale so the final resized image stays within budget after converting to integer pixel dimensions and computing patch coverage.

```
shrink_factor = sqrt((32^2 * patch_budget) / (width * height))
adjusted_shrink_factor = shrink_factor * min(
  floor(width * shrink_factor / 32) / (width * shrink_factor / 32),
  floor(height * shrink_factor / 32) / (height * shrink_factor / 32)
)
```

C. Convert the adjusted scale into integer pixel dimensions, then compute the number of patches needed to cover the resized image. This resized patch count is the image-token count before applying the model multiplier, and it is capped by the model's patch budget.

```
resized_patch_count = ceil(resized_width/32)×ceil(resized_height/32)
```

D. Apply a multiplier based on the model to get the total tokens:

| Model           | Multiplier |
| --------------- | ---------- |
| `gpt-5-mini`    | 1.62       |
| `gpt-5-nano`    | 2.46       |
| `gpt-4.1-mini*` | 1.62       |
| `gpt-4.1-nano*` | 2.46       |
| `o4-mini`       | 1.72       |

_For `gpt-4.1-mini` and `gpt-4.1-nano`, this applies to the 2025-04-14 snapshot variants._

**Cost calculation examples for a model with a 1,536-patch budget**

- A 1024 x 1024 image has a post-resize patch count of **1024**
  - A. `original_patch_count = ceil(1024 / 32) * ceil(1024 / 32) = 32 * 32 = 1024`
  - B. `1024` is below the `1,536` patch budget, so no resize is needed.
  - C. `resized_patch_count = 1024`
  - Resized patch count before the model multiplier: `1024`
  - Multiply by the model's token multiplier to get the billed token units.
- A 1800 x 2400 image has a post-resize patch count of **1452**
  - A. `original_patch_count = ceil(1800 / 32) * ceil(2400 / 32) = 57 * 75 = 4275`
  - B. `4275` exceeds the `1,536` patch budget, so we first compute `shrink_factor = sqrt((32^2 * 1536) / (1800 * 2400)) = 0.603`.
  - We then adjust that scale so the final integer pixel dimensions stay within budget after patch counting: `adjusted_shrink_factor = 0.603 * min(floor(1800 * 0.603 / 32) / (1800 * 0.603 / 32), floor(2400 * 0.603 / 32) / (2400 * 0.603 / 32)) = 0.586`.
  - Resized image in integer pixels: `1056 x 1408`
  - C. `resized_patch_count = ceil(1056 / 32) * ceil(1408 / 32) = 33 * 44 = 1452`
  - Resized patch count before the model multiplier: `1452`
  - Multiply by the model's token multiplier to get the billed token units.

### Tile-based image tokenization

#### GPT-4o, GPT-4.1, GPT-4o-mini, CUA, and o-series (except o4-mini)

The token cost of an image is determined by two factors: size and detail.

Any image with `"detail": "low"` costs a set, base number of tokens. This amount varies by model. To calculate the cost of an image with `"detail": "high"`, we do the following:

- Scale to fit in a 2048px x 2048px square, maintaining original aspect ratio
- Scale so that the image's shortest side is 768px long
- Count the number of 512px squares in the image. Each square costs a set amount of tokens, shown below.
- Add the base tokens to the total

| Model                    | Base tokens | Tile tokens |
| ------------------------ | ----------- | ----------- |
| gpt-5, gpt-5-chat-latest | 70          | 140         |
| 4o, 4.1, 4.5             | 85          | 170         |
| 4o-mini                  | 2833        | 5667        |
| o1, o1-pro, o3           | 75          | 150         |
| computer-use-preview     | 65          | 129         |

### GPT Image 1

For GPT Image 1, we calculate the cost of an image input the same way as described above, except that we scale down the image so that the shortest side is 512px instead of 768px.
The price depends on the dimensions of the image and the [input fidelity](https://developers.openai.com/api/docs/guides/image-generation?image-generation-model=gpt-image-1#input-fidelity).

When input fidelity is set to low, the base cost is 65 image tokens, and each tile costs 129 image tokens.
When using high input fidelity, we add a set number of tokens based on the image's aspect ratio in addition to the image tokens described above.

- If your image is square, we add 4160 extra input image tokens.
- If it is closer to portrait or landscape, we add 6240 extra tokens.

To see pricing for image input tokens, refer to our [pricing page](https://developers.openai.com/api/docs/pricing#latest-models).

## Limitations

While models with vision capabilities are powerful and can be used in many situations, it's important to understand the limitations of these models. Here are some known limitations:

- **Medical images**: The model is not suitable for interpreting specialized medical images like CT scans and shouldn't be used for medical advice.
- **Non-English**: The model may not perform optimally when handling images with text of non-Latin alphabets, such as Japanese or Korean.
- **Small text**: Enlarge text within the image to improve readability. When available, using `"detail": "original"` can also help performance.
- **Rotation**: The model may misinterpret rotated or upside-down text and images.
- **Visual elements**: The model may struggle to understand graphs or text where colors or styles—like solid, dashed, or dotted lines—vary.
- **Spatial reasoning**: The model struggles with tasks requiring precise spatial localization, such as identifying chess positions.
- **Accuracy**: The model may generate incorrect descriptions or captions in certain scenarios.
- **Image shape**: The model struggles with panoramic and fisheye images.
- **Metadata and resizing**: The model doesn't process original file names or metadata. Depending on image size and `detail` level, images may be resized before analysis, affecting their original dimensions.
- **Counting**: The model may give approximate counts for objects in images.
- **CAPTCHAS**: For safety reasons, our system blocks the submission of CAPTCHAs.

---

We process images at the token level, so each image we process counts towards your tokens per minute (TPM) limit.

For the most precise and up-to-date estimates for image processing, please use our image pricing calculator available [here](https://openai.com/api/pricing/).

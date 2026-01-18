Gemini models are built to be multimodal from the ground up, unlocking a wide range of image processing and computer vision tasks including but not limited to image captioning, classification, and visual question answering without having to train specialized ML models.
| **Tip:** In addition to their general multimodal capabilities, Gemini models (2.0 and newer) offer **improved accuracy** for specific use cases like [object detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection) and [segmentation](https://ai.google.dev/gemini-api/docs/image-understanding#segmentation), through additional training. See the [Capabilities](https://ai.google.dev/gemini-api/docs/image-understanding#capabilities) section for more details.

## Passing images to Gemini

You can provide images as input to Gemini using two methods:

- [Passing inline image data](https://ai.google.dev/gemini-api/docs/image-understanding#inline-image): Ideal for smaller files (total request size less than 20MB, including prompts).
- [Uploading images using the File API](https://ai.google.dev/gemini-api/docs/image-understanding#upload-image): Recommended for larger files or for reusing images across multiple requests.

### Passing inline image data

You can pass inline image data in the
request to `generateContent`. You can provide image data as Base64 encoded
strings or by reading local files directly (depending on the language).

The following example shows how to read an image from a local file and pass
it to `generateContent` API for processing.  

### Python

      from google import genai
      from google.genai import types

      with open('path/to/small-sample.jpg', 'rb') as f:
          image_bytes = f.read()

      client = genai.Client()
      response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[
          types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
          ),
          'Caption this image.'
        ]
      )

      print(response.text)

### JavaScript

    import { GoogleGenAI } from "@google/genai";
    import * as fs from "node:fs";

    const ai = new GoogleGenAI({});
    const base64ImageFile = fs.readFileSync("path/to/small-sample.jpg", {
      encoding: "base64",
    });

    const contents = [
      {
        inlineData: {
          mimeType: "image/jpeg",
          data: base64ImageFile,
        },
      },
      { text: "Caption this image." },
    ];

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: contents,
    });
    console.log(response.text);

### Go

    bytes, _ := os.ReadFile("path/to/small-sample.jpg")

    parts := []*genai.Part{
      genai.NewPartFromBytes(bytes, "image/jpeg"),
      genai.NewPartFromText("Caption this image."),
    }

    contents := []*genai.Content{
      genai.NewContentFromParts(parts, genai.RoleUser),
    }

    result, _ := client.Models.GenerateContent(
      ctx,
      "gemini-3-flash-preview",
      contents,
      nil,
    )

    fmt.Println(result.Text())

### REST

    IMG_PATH="/path/to/your/image1.jpg"

    if [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
    B64FLAGS="--input"
    else
    B64FLAGS="-w0"
    fi

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
        "contents": [{
        "parts":[
            {
                "inline_data": {
                "mime_type":"image/jpeg",
                "data": "'"$(base64 $B64FLAGS $IMG_PATH)"'"
                }
            },
            {"text": "Caption this image."},
        ]
        }]
    }' 2> /dev/null

You can also fetch an image from a URL, convert it to bytes, and pass it to
`generateContent` as shown in the following examples.  

### Python

    from google import genai
    from google.genai import types

    import requests

    image_path = "https://goo.gle/instrument-img"
    image_bytes = requests.get(image_path).content
    image = types.Part.from_bytes(
      data=image_bytes, mime_type="image/jpeg"
    )

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=["What is this image?", image],
    )

    print(response.text)

### JavaScript

    import { GoogleGenAI } from "@google/genai";

    async function main() {
      const ai = new GoogleGenAI({});

      const imageUrl = "https://goo.gle/instrument-img";

      const response = await fetch(imageUrl);
      const imageArrayBuffer = await response.arrayBuffer();
      const base64ImageData = Buffer.from(imageArrayBuffer).toString('base64');

      const result = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [
        {
          inlineData: {
            mimeType: 'image/jpeg',
            data: base64ImageData,
          },
        },
        { text: "Caption this image." }
      ],
      });
      console.log(result.text);
    }

    main();

### Go

    package main

    import (
      "context"
      "fmt"
      "os"
      "io"
      "net/http"
      "google.golang.org/genai"
    )

    func main() {
      ctx := context.Background()
      client, err := genai.NewClient(ctx, nil)
      if err != nil {
          log.Fatal(err)
      }

      // Download the image.
      imageResp, _ := http.Get("https://goo.gle/instrument-img")

      imageBytes, _ := io.ReadAll(imageResp.Body)

      parts := []*genai.Part{
        genai.NewPartFromBytes(imageBytes, "image/jpeg"),
        genai.NewPartFromText("Caption this image."),
      }

      contents := []*genai.Content{
        genai.NewContentFromParts(parts, genai.RoleUser),
      }

      result, _ := client.Models.GenerateContent(
        ctx,
        "gemini-3-flash-preview",
        contents,
        nil,
      )

      fmt.Println(result.Text())
    }

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
          "contents": [{
            "parts":[
                {
                  "inline_data": {
                    "mime_type":"'"$MIME_TYPE"'",
                    "data": "'"$IMAGE_B64"'"
                  }
                },
                {"text": "Caption this image."}
            ]
          }]
        }' 2> /dev/null

| **Note:** Inline image data limits your total request size (text prompts, system instructions, and inline bytes) to 20MB. For larger requests, [upload image files](https://ai.google.dev/gemini-api/docs/image-understanding#upload-image) using the File API. Files API is also more efficient for scenarios that use the same image repeatedly.

### Uploading images using the File API

For large files or to be able to use the same image file repeatedly, use the
Files API. The following code uploads an image file and then uses the file in a
call to `generateContent`. See the [Files API guide](https://ai.google.dev/gemini-api/docs/files) for
more information and examples.  

### Python

    from google import genai

    client = genai.Client()

    my_file = client.files.upload(file="path/to/sample.jpg")

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[my_file, "Caption this image."],
    )

    print(response.text)

### JavaScript

    import {
      GoogleGenAI,
      createUserContent,
      createPartFromUri,
    } from "@google/genai";

    const ai = new GoogleGenAI({});

    async function main() {
      const myfile = await ai.files.upload({
        file: "path/to/sample.jpg",
        config: { mimeType: "image/jpeg" },
      });

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: createUserContent([
          createPartFromUri(myfile.uri, myfile.mimeType),
          "Caption this image.",
        ]),
      });
      console.log(response.text);
    }

    await main();

### Go

    package main

    import (
      "context"
      "fmt"
      "os"
      "google.golang.org/genai"
    )

    func main() {
      ctx := context.Background()
      client, err := genai.NewClient(ctx, nil)
      if err != nil {
          log.Fatal(err)
      }

      uploadedFile, _ := client.Files.UploadFromPath(ctx, "path/to/sample.jpg", nil)

      parts := []*genai.Part{
          genai.NewPartFromText("Caption this image."),
          genai.NewPartFromURI(uploadedFile.URI, uploadedFile.MIMEType),
      }

      contents := []*genai.Content{
          genai.NewContentFromParts(parts, genai.RoleUser),
      }

      result, _ := client.Models.GenerateContent(
          ctx,
          "gemini-3-flash-preview",
          contents,
          nil,
      )

      fmt.Println(result.Text())
    }

### REST

    IMAGE_PATH="path/to/sample.jpg"
    MIME_TYPE=$(file -b --mime-type "${IMAGE_PATH}")
    NUM_BYTES=$(wc -c < "${IMAGE_PATH}")
    DISPLAY_NAME=IMAGE

    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -D upload-header.tmp \
      -H "X-Goog-Upload-Protocol: resumable" \
      -H "X-Goog-Upload-Command: start" \
      -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
      -H "Content-Type: application/json" \
      -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

    upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file}"

    # Upload the actual bytes.
    curl "${upload_url}" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H "Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Offset: 0" \
      -H "X-Goog-Upload-Command: upload, finalize" \
      --data-binary "@${IMAGE_PATH}" 2> /dev/null > file_info.json

    file_uri=$(jq -r ".file.uri" file_info.json)
    echo file_uri=$file_uri

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
        -H "x-goog-api-key: $GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"file_data":{"mime_type": "'"${MIME_TYPE}"'", "file_uri": "'"${file_uri}"'"}},
              {"text": "Caption this image."}]
            }]
          }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

## Prompting with multiple images

You can provide multiple images in a single prompt by including multiple image
`Part` objects in the `contents` array. These can be a mix of inline data
(local files or URLs) and File API references.  

### Python

    from google import genai
    from google.genai import types

    client = genai.Client()

    # Upload the first image
    image1_path = "path/to/image1.jpg"
    uploaded_file = client.files.upload(file=image1_path)

    # Prepare the second image as inline data
    image2_path = "path/to/image2.png"
    with open(image2_path, 'rb') as f:
        img2_bytes = f.read()

    # Create the prompt with text and multiple images
    response = client.models.generate_content(

        model="gemini-3-flash-preview",
        contents=[
            "What is different between these two images?",
            uploaded_file,  # Use the uploaded file reference
            types.Part.from_bytes(
                data=img2_bytes,
                mime_type='image/png'
            )
        ]
    )

    print(response.text)

### JavaScript

    import {
      GoogleGenAI,
      createUserContent,
      createPartFromUri,
    } from "@google/genai";
    import * as fs from "node:fs";

    const ai = new GoogleGenAI({});

    async function main() {
      // Upload the first image
      const image1_path = "path/to/image1.jpg";
      const uploadedFile = await ai.files.upload({
        file: image1_path,
        config: { mimeType: "image/jpeg" },
      });

      // Prepare the second image as inline data
      const image2_path = "path/to/image2.png";
      const base64Image2File = fs.readFileSync(image2_path, {
        encoding: "base64",
      });

      // Create the prompt with text and multiple images

      const response = await ai.models.generateContent({

        model: "gemini-3-flash-preview",
        contents: createUserContent([
          "What is different between these two images?",
          createPartFromUri(uploadedFile.uri, uploadedFile.mimeType),
          {
            inlineData: {
              mimeType: "image/png",
              data: base64Image2File,
            },
          },
        ]),
      });
      console.log(response.text);
    }

    await main();

### Go

    // Upload the first image
    image1Path := "path/to/image1.jpg"
    uploadedFile, _ := client.Files.UploadFromPath(ctx, image1Path, nil)

    // Prepare the second image as inline data
    image2Path := "path/to/image2.jpeg"
    imgBytes, _ := os.ReadFile(image2Path)

    parts := []*genai.Part{
      genai.NewPartFromText("What is different between these two images?"),
      genai.NewPartFromBytes(imgBytes, "image/jpeg"),
      genai.NewPartFromURI(uploadedFile.URI, uploadedFile.MIMEType),
    }

    contents := []*genai.Content{
      genai.NewContentFromParts(parts, genai.RoleUser),
    }

    result, _ := client.Models.GenerateContent(
      ctx,
      "gemini-3-flash-preview",
      contents,
      nil,
    )

    fmt.Println(result.Text())

### REST

    # Upload the first image
    IMAGE1_PATH="path/to/image1.jpg"
    MIME1_TYPE=$(file -b --mime-type "${IMAGE1_PATH}")
    NUM1_BYTES=$(wc -c < "${IMAGE1_PATH}")
    DISPLAY_NAME1=IMAGE1

    tmp_header_file1=upload-header1.tmp

    curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -D upload-header1.tmp \
      -H "X-Goog-Upload-Protocol: resumable" \
      -H "X-Goog-Upload-Command: start" \
      -H "X-Goog-Upload-Header-Content-Length: ${NUM1_BYTES}" \
      -H "X-Goog-Upload-Header-Content-Type: ${MIME1_TYPE}" \
      -H "Content-Type: application/json" \
      -d "{'file': {'display_name': '${DISPLAY_NAME1}'}}" 2> /dev/null

    upload_url1=$(grep -i "x-goog-upload-url: " "${tmp_header_file1}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file1}"

    curl "${upload_url1}" \
      -H "Content-Length: ${NUM1_BYTES}" \
      -H "X-Goog-Upload-Offset: 0" \
      -H "X-Goog-Upload-Command: upload, finalize" \
      --data-binary "@${IMAGE1_PATH}" 2> /dev/null > file_info1.json

    file1_uri=$(jq ".file.uri" file_info1.json)
    echo file1_uri=$file1_uri

    # Prepare the second image (inline)
    IMAGE2_PATH="path/to/image2.png"
    MIME2_TYPE=$(file -b --mime-type "${IMAGE2_PATH}")

    if [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
      B64FLAGS="--input"
    else
      B64FLAGS="-w0"
    fi
    IMAGE2_BASE64=$(base64 $B64FLAGS $IMAGE2_PATH)

    # Now generate content using both images
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent" \
        -H "x-goog-api-key: $GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "What is different between these two images?"},
              {"file_data":{"mime_type": "'"${MIME1_TYPE}"'", "file_uri": '$file1_uri'}},
              {
                "inline_data": {
                  "mime_type":"'"${MIME2_TYPE}"'",
                  "data": "'"$IMAGE2_BASE64"'"
                }
              }
            ]
          }]
        }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

## Object detection

From Gemini 2.0 onwards, models are further trained to detect objects in an
image and get their bounding box coordinates. The coordinates, relative to image
dimensions, scale to \[0, 1000\]. You need to descale these coordinates based on
your original image size.  

### Python

    from google import genai
    from google.genai import types
    from PIL import Image
    import json

    client = genai.Client()
    prompt = "Detect the all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."

    image = Image.open("/path/to/image.png")

    config = types.GenerateContentConfig(
      response_mime_type="application/json"
      )

    response = client.models.generate_content(model="gemini-3-flash-preview",
                                              contents=[image, prompt],
                                              config=config
                                              )

    width, height = image.size
    bounding_boxes = json.loads(response.text)

    converted_bounding_boxes = []
    for bounding_box in bounding_boxes:
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
        converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

    print("Image size: ", width, height)
    print("Bounding boxes:", converted_bounding_boxes)

| **Note:** The model also supports generating bounding boxes based on custom instructions, such as: "Show bounding boxes of all green objects in this image". It also support custom labels like "label the items with the allergens they can contain".

For more examples, check following notebooks in the [Gemini Cookbook](https://github.com/google-gemini/cookbook):

- [2D spatial understanding notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb)
- [Experimental 3D pointing notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Spatial_understanding_3d.ipynb)

## Segmentation

Starting with Gemini 2.5, models not only detect items but also segment them
and provide their contour masks.

The model predicts a JSON list, where each item represents a segmentation mask.
Each item has a bounding box ("`box_2d`") in the format `[y0, x0, y1, x1]` with
normalized coordinates between 0 and 1000, a label ("`label`") that identifies
the object, and finally the segmentation mask inside the bounding box, as base64
encoded png that is a probability map with values between 0 and 255.
The mask needs to be resized to match the bounding box dimensions, then
binarized at your confidence threshold (127 for the midpoint).
**Note:** For better results, disable [thinking](https://ai.google.dev/gemini-api/docs/thinking) by setting the thinking budget to 0. See code sample below for an example.  

### Python

    from google import genai
    from google.genai import types
    from PIL import Image, ImageDraw
    import io
    import base64
    import json
    import numpy as np
    import os

    client = genai.Client()

    def parse_json(json_output: str):
      # Parsing out the markdown fencing
      lines = json_output.splitlines()
      for i, line in enumerate(lines):
        if line == "```json":
          json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
          output = json_output.split("```")[0]  # Remove everything after the closing "```"
          break  # Exit the loop once "```json" is found
      return json_output

    def extract_segmentation_masks(image_path: str, output_dir: str = "segmentation_outputs"):
      # Load and resize image
      im = Image.open(image_path)
      im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

      prompt = """
      Give the segmentation masks for the wooden and glass items.
      Output a JSON list of segmentation masks where each entry contains the 2D
      bounding box in the key "box_2d", the segmentation mask in key "mask", and
      the text label in the key "label". Use descriptive labels.
      """

      config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # set thinking_budget to 0 for better results in object detection
      )

      response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, im], # Pillow images can be directly passed as inputs (which will be converted by the SDK)
        config=config
      )

      # Parse JSON response
      items = json.loads(parse_json(response.text))

      # Create output directory
      os.makedirs(output_dir, exist_ok=True)

      # Process each mask
      for i, item in enumerate(items):
          # Get bounding box coordinates
          box = item["box_2d"]
          y0 = int(box[0] / 1000 * im.size[1])
          x0 = int(box[1] / 1000 * im.size[0])
          y1 = int(box[2] / 1000 * im.size[1])
          x1 = int(box[3] / 1000 * im.size[0])

          # Skip invalid boxes
          if y0 >= y1 or x0 >= x1:
              continue

          # Process mask
          png_str = item["mask"]
          if not png_str.startswith("data:image/png;base64,"):
              continue

          # Remove prefix
          png_str = png_str.removeprefix("data:image/png;base64,")
          mask_data = base64.b64decode(png_str)
          mask = Image.open(io.BytesIO(mask_data))

          # Resize mask to match bounding box
          mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

          # Convert mask to numpy array for processing
          mask_array = np.array(mask)

          # Create overlay for this mask
          overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
          overlay_draw = ImageDraw.Draw(overlay)

          # Create overlay for the mask
          color = (255, 255, 255, 200)
          for y in range(y0, y1):
              for x in range(x0, x1):
                  if mask_array[y - y0, x - x0] > 128:  # Threshold for mask
                      overlay_draw.point((x, y), fill=color)

          # Save individual mask and its overlay
          mask_filename = f"{item['label']}_{i}_mask.png"
          overlay_filename = f"{item['label']}_{i}_overlay.png"

          mask.save(os.path.join(output_dir, mask_filename))

          # Create and save overlay
          composite = Image.alpha_composite(im.convert('RGBA'), overlay)
          composite.save(os.path.join(output_dir, overlay_filename))
          print(f"Saved mask and overlay for {item['label']} to {output_dir}")

    # Example usage
    if __name__ == "__main__":
      extract_segmentation_masks("path/to/image.png")

Check the
[segmentation example](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb#scrollTo=WQJTJ8wdGOKx)
in the cookbook guide for a more detailed example.
![A table with cupcakes, with the wooden and glass objects highlighted](https://ai.google.dev/static/gemini-api/docs/images/segmentation.jpg) An example segmentation output with objects and segmentation masks

## Supported image formats

Gemini supports the following image format MIME types:

- PNG - `image/png`
- JPEG - `image/jpeg`
- WEBP - `image/webp`
- HEIC - `image/heic`
- HEIF - `image/heif`

To learn about other file input methods, see the
[File input methods](https://ai.google.dev/gemini-api/docs/file-input-methods) guide.

## Capabilities

All Gemini model versions are multimodal and can be utilized in a wide range of
image processing and computer vision tasks including but not limited to image captioning,
visual question and answering, image classification, object detection and segmentation.

Gemini can reduce the need to use specialized ML models depending on your quality and performance requirements.

Some later model versions are specifically trained improve accuracy of specialized tasks in addition to generic capabilities:

- **Gemini 2.0 models** are further trained to support enhanced [object detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection).

- **Gemini 2.5 models** are further trained to support enhanced [segmentation](https://ai.google.dev/gemini-api/docs/image-understanding#segmentation) in addition to [object detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection).

## Limitations and key technical information

### File limit

Gemini 2.5 Pro/Flash, 2.0 Flash, 1.5 Pro, and 1.5 Flash support a
maximum of 3,600 image files per request.

### Token calculation

- **Gemini 1.5 Flash and Gemini 1.5 Pro**: 258 tokens if both dimensions \<= 384 pixels. Larger images are tiled (min tile 256px, max 768px, resized to 768x768), with each tile costing 258 tokens.
- **Gemini 2.0 Flash and Gemini 2.5 Flash/Pro**: 258 tokens if both dimensions \<= 384 pixels. Larger images are tiled into 768x768 pixel tiles, each costing 258 tokens.

A rough formula for calculating the number of tiles is as follows:

- Calculate the crop unit size which is roughly: floor(min(width, height) / 1.5).
- Divide each dimension by the crop unit size and multiply together to get the number of tiles.

For example, for an image of dimensions 960x540 would have a crop unit size
of 360. Divide each dimension by 360 and the number of tile is 3 \* 2 = 6.

### Media resolution

Gemini 3 introduces granular control over multimodal vision processing with the
`media_resolution` parameter. The `media_resolution` parameter determines the
**maximum number of tokens allocated per input image or video frame.**
Higher resolutions improve the model's ability to
read fine text or identify small details, but increase token usage and latency.

For more details about the parameter and how it can impact token calculations,
see the [media resolution](https://ai.google.dev/gemini-api/docs/media-resolution) guide.

## Tips and best practices

- Verify that images are correctly rotated.
- Use clear, non-blurry images.
- When using a single image with text, place the text prompt *after* the image part in the `contents` array.

## What's next

This guide shows you how to upload image files and generate text outputs from image
inputs. To learn more, see the following resources:

- [Files API](https://ai.google.dev/gemini-api/docs/files): Learn more about uploading and managing files for use with Gemini.
- [System instructions](https://ai.google.dev/gemini-api/docs/text-generation#system-instructions): System instructions let you steer the behavior of the model based on your specific needs and use cases.
- [File prompting strategies](https://ai.google.dev/gemini-api/docs/files#prompt-guide): The Gemini API supports prompting with text, image, audio, and video data, also known as multimodal prompting.
- [Safety guidance](https://ai.google.dev/gemini-api/docs/safety-guidance): Sometimes generative AI models produce unexpected outputs, such as outputs that are inaccurate, biased, or offensive. Post-processing and human evaluation are essential to limit the risk of harm from such outputs.

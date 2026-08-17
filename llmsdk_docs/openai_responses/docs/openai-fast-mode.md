# Fast mode

> For the complete documentation index, see [llms.txt](/llms.txt). Markdown versions of documentation pages are available by appending `.md` to the page URL.

Fast mode delivers up to 2.5× faster speeds and more consistent latency while keeping pay-as-you-go flexibility. Fast mode is ideal for high-value, user-facing applications with regular traffic where latency is paramount.

Priority processing was renamed Fast mode on July 30, 2026. We also increased
  the speed at which Fast mode operates for `gpt-5.6-sol` to make it up to 2.5×
  faster than Standard processing. You can use either `service_tier: "priority"`
  or `service_tier: "fast"` in your API requests to access this functionality.

## Configuring Fast mode

You can configure requests to the Responses API or Chat Completions API to use Fast mode through either a request parameter or a project setting.

To opt in to Fast mode for an individual request, set the [`service_tier` parameter](https://platform.openai.com/docs/api-reference/responses/create#responses-create-service_tier) to `fast`. Setting `service_tier` to `priority` provides the same behavior for supported models.

Create a response with Fast mode

```javascript
import OpenAI from "openai";

const openai = new OpenAI();

const response = await openai.responses.create({
  model: "gpt-5.6-sol",
  input: "What does 'fit check for my napalm era' mean?",
  service_tier: "fast",
});

console.log(response);
```

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-sol",
    input="What does 'fit check for my napalm era' mean?",
    service_tier="fast",
)
print(response)
```

```go
package main

import (
	"context"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

func main() {
	client := openai.NewClient()
	response, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model:       "gpt-5.6-sol",
		ServiceTier: "fast",
		Input:       responses.ResponseNewParamsInputUnion{OfString: openai.String("What does 'fit check for my napalm era' mean?")},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(response.OutputText())
}
```

```ruby
require "openai"

client = OpenAI::Client.new

response = client.responses.create(
  model: "gpt-5.6-sol",
  service_tier: :fast,
  input: "What does 'fit check for my napalm era' mean?"
)

puts(response.output_text)
```

```bash
curl https://api.openai.com/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.6-sol",
    "input": "What does 'fit check for my napalm era' mean?",
    "service_tier": "fast"
  }'
```


To opt in at the project level, open **Settings**, select **General** under **Project**, and change **Project Service Tier** to **Fast**. Requests that don't specify a `service_tier` then default to Fast mode. Requests for the project transition gradually to Fast mode over time.

The `service_tier` field in the [Responses](https://platform.openai.com/docs/api-reference/responses/object#responses/object-service_tier) or [Chat Completions](https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier) response object identifies the tier used to process the request. For GPT-5.6 and earlier models, the response returns `priority` whether the request specifies `priority` or `fast`.

## Rate limits and ramp rate

**Baseline limits**

Fast mode consumption counts toward rate limits the same way as Standard processing. Use your usual retry logic and wait between attempts. For a given model, Standard processing and Fast mode share the same rate limit.

**Ramp rate limit**

If your traffic ramps too fast, the system may downgrade some Fast mode requests to standard speeds and charge standard rates. When this happens, the response contains `service_tier: "default"`. The ramp rate limit may apply if you send at least 1 million tokens per minute (TPM) and increase TPM by more than 50% within 15 minutes.

To avoid triggering the ramp rate limit:

- Ramp gradually when changing models or snapshots.
- Use feature flags to shift traffic over hours, not instantly.
- Avoid running large extract, transform, and load (ETL) or batch jobs in Fast mode.

## Usage considerations

- Fast mode charges a per-token premium over Standard processing. See the [pricing page](https://developers.openai.com/api/docs/pricing?latest-pricing=fast) for details and supported models.
- Cached input discounts still apply to Fast mode requests.
- Fast mode supports multimodal requests, including image inputs.
- To view Fast mode requests in the usage dashboard, select the option to group by service tier. For GPT-5.6 and earlier models, these requests appear as `priority` even when you specify `fast`.
- GPT-5.6 models support long context. Fast mode doesn't support fine-tuned models or embeddings.

## Frequently asked questions

For account and policy information, see the [Fast mode FAQ](https://help.openai.com/en/articles/11647665-priority-processing-faq).

### Is Fast mode available in all regions?

Availability depends on the laws and regulations in each jurisdiction. Contact your account director if you have questions about availability in your region.

### How does Fast mode interact with Scale Tier?

Scale Tier and Fast mode are separate. Fast mode requests have separate billing and don't count against purchased Scale Tier TPM bundles. Scale Tier spillover traffic doesn't automatically move to Fast mode.

### How is Fast mode billed?

Fast mode charges a per-token premium compared with Standard processing. All processing modes count toward your annual Enterprise spend commitment, and eligible cached input tokens receive the same discounts available for Standard processing.

To review usage, open the usage dashboard, select Responses or Chat Completions, and group by service tier. To review costs, group by line item.

### Which models and modalities support Fast mode?

Fast mode supports the multimodal capabilities available with Standard processing, including image inputs. GPT-5.6 models support long context. Fast mode doesn't support fine-tuned models or embeddings. Future GPT models may support Fast mode, but support isn't guaranteed for every model.

### Are ramp rate limits shared across projects or organizations?

Yes. All your traffic contributes to the same ramp rate limit. If you routinely encounter ramp rate limits, consider purchasing Scale Tier quota.

### What happens if Fast mode doesn't meet its latency target?

Contact your account director if you have questions or concerns. Fast mode and Scale Tier receive the same service-level agreement treatment, and eligible Enterprise agreements may provide service credits when those targets aren't met.

### Is Fast mode compatible with data residency, Zero Data Retention, and a BAA?

Yes. Fast mode is compatible with data residency, Zero Data Retention, and a Business Associate Agreement (BAA). Existing endpoint, tool, eligibility, and contractual requirements still apply. See the [Your data guide](https://developers.openai.com/api/docs/guides/your-data) for details.
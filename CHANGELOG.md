# Changelog

- [26/03/11] GPT-5.4 is supported. We now add `phase` labels to assistant messages, and preserve and send them to the server.

- [26/03/04] Claude 4.6 is supported. We switch to using the adaptive thinking and `effort` parameter instead of the thinking budget.

- [26/02/26] Supports Claude on Amazon Bedrock. Bedrock requires image base64 encoding, we convert images to base64 in the client.

- [26/02/15] Fix encrypted thinking message in Claude models. It needs to be preserved and sent to the server.

- [26/02/15] Fix the calculation of token usage in from OpenRouter provider.

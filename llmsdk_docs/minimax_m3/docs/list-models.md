> ## Documentation Index
> Fetch the complete documentation index at: https://platform.minimax.io/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# List Models

> Returns a list of all available models compatible with OpenAI API specification.



## OpenAPI

````yaml api-reference/models/openai/api/list-models.json GET /v1/models
openapi: 3.1.0
info:
  title: MiniMax Models API
  description: MiniMax models API compatible with OpenAI API specification.
  version: 1.0.0
servers:
  - url: https://api.minimax.io
security:
  - bearerAuth: []
paths:
  /v1/models:
    get:
      tags:
        - Models
      summary: List Models
      description: >-
        Returns a list of all available models. This endpoint is compatible with
        OpenAI API specification.
      operationId: listModels
      responses:
        '200':
          description: A list of available models.
          content:
            application/json:
              schema:
                type: object
                properties:
                  object:
                    type: string
                    description: Object type, always "list"
                  data:
                    type: array
                    description: Array of model objects
                    items:
                      type: object
                      properties:
                        id:
                          type: string
                          description: Model identifier
                        object:
                          type: string
                          description: Object type, always "model"
                        created:
                          type: integer
                          description: Unix timestamp when the model was created
                        owned_by:
                          type: string
                          description: Organization that owns the model
              examples:
                Default:
                  value:
                    object: list
                    data:
                      - id: MiniMax-M3
                        object: model
                        created: 1780272000
                        owned_by: minimax
                      - id: MiniMax-M2.7
                        object: model
                        created: 1773799200
                        owned_by: minimax
                      - id: MiniMax-M2.5
                        object: model
                        created: 1770948000
                        owned_by: minimax
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |-
        `HTTP: Bearer Auth`
         - Security Scheme Type: http
         - HTTP Authorization Scheme: Bearer API_key, used for account verification, can be viewed in [Account Management > API Keys](https://platform.minimax.io/user-center/basic-information/interface-key)

````
// Copyright 2025 Prism Shadow. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Example demonstrating the new object-based API for AutoLLMClient.
 *
 * This example shows how to use the new constructor syntax that accepts
 * a configuration object instead of multiple parameters.
 */

import { AutoLLMClient } from "../src/autoClient";

console.log("=".repeat(60));
console.log("AgentHub Object-based API Example");
console.log("=".repeat(60));

// Example 1: Basic usage with just model name
console.log("\n1. Basic usage - just model name:");
console.log("   const client = new AutoLLMClient({ model: 'gemini-3-flash-preview' });");
// const client1 = new AutoLLMClient({ model: 'gemini-3-flash-preview' });

// Example 2: With API key
console.log("\n2. With API key:");
console.log("   const client = new AutoLLMClient({");
console.log("     model: 'claude-sonnet-4-5-20250929',");
console.log("     apiKey: 'your-api-key'");
console.log("   });");
// const client2 = new AutoLLMClient({
//   model: 'claude-sonnet-4-5-20250929',
//   apiKey: 'your-api-key'
// });

// Example 3: With all options
console.log("\n3. With all configuration options:");
console.log("   const client = new AutoLLMClient({");
console.log("     model: 'gpt-5.2',");
console.log("     apiKey: 'your-api-key',");
console.log("     baseUrl: 'https://api.openai.com',");
console.log("     clientType: 'gpt-5.2'");
console.log("   });");
// const client3 = new AutoLLMClient({
//   model: 'gpt-5.2',
//   apiKey: 'your-api-key',
//   baseUrl: 'https://api.openai.com',
//   clientType: 'gpt-5.2'
// });

console.log("\n" + "=".repeat(60));
console.log("Benefits of object-based API:");
console.log("- More extensible - easy to add new parameters in the future");
console.log("- Self-documenting - parameter names are explicit");
console.log("- Optional parameters are clearer");
console.log("- Better IDE autocomplete support");
console.log("=".repeat(60));

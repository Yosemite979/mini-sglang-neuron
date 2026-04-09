# Tool Calling Test Results

Tested with **Qwen/Qwen3-4B** on **trn1.2xlarge** (TP=2) via LiteLLM gateway.

## 1. Regular Chat (no tools) ✅

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "max_tokens": 50
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-fb7512e1c1c4",
  "model": "Qwen/Qwen3-4B",
  "object": "chat.completion",
  "choices": [{
    "finish_reason": "stop",
    "index": 0,
    "message": {
      "content": "<think>\nOkay, the user is asking \"What is 2+2? Answer in one word.\" ...\n</think>\n\nFour.",
      "role": "assistant"
    }
  }]
}
```

## 2. Tool Calling — Non-Streaming ✅

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Calculate 123 * 456"}],
    "max_tokens": 200,
    "stream": false,
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Perform arithmetic calculations",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
          },
          "required": ["expression"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-d3304d5cfef9",
  "model": "Qwen/Qwen3-4B",
  "object": "chat.completion",
  "choices": [{
    "finish_reason": "tool_calls",
    "index": 0,
    "message": {
      "content": "",
      "role": "assistant",
      "tool_calls": [{
        "index": 0,
        "function": {
          "arguments": "{\"expression\": \"123 * 456\"}",
          "name": "calculator"
        },
        "id": "call_ec84438c",
        "type": "function"
      }],
      "reasoning_content": "\nOkay, the user wants me to calculate 123 multiplied by 456. Let me check the tools available. There's a calculator function that takes an expression as a string. So I need to call that function with the expression \"123 * 456\".\n"
    }
  }]
}
```

## 3. Tool Calling — Streaming ✅

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "What is 99 + 1?"}],
    "max_tokens": 200,
    "stream": true,
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Perform arithmetic calculations",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
          },
          "required": ["expression"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

**Response (SSE chunks):**
```
data: {"id":"chatcmpl-334089d6ada3","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"<think>\nOkay, the user is asking what 99 plus 1 is...","role":"assistant","tool_calls":[{"id":"call_23ff6198","function":{"arguments":"{\"expression\": \"99 + 1\"}","name":"calculator"},"type":"function","index":0}]}}]}

data: {"id":"chatcmpl-334089d6ada3","object":"chat.completion.chunk","choices":[{"finish_reason":"tool_calls","index":0,"delta":{"content":"..."}}]}

data: [DONE]
```

## Summary

| Test | Stream | Tools | Result |
|------|--------|-------|--------|
| Regular chat | No | No | ✅ Returns thinking + answer |
| Tool calling | No | Yes | ✅ `finish_reason: "tool_calls"`, correct function args |
| Tool calling | Yes | Yes | ✅ SSE chunks with tool_calls, `finish_reason: "tool_calls"` |

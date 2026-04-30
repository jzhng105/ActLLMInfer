# ActLLMInfer

Integrated LLM inference engine with a **LangChain-Core-style** interface. One small package, one consistent API across the major Chinese and US chat-LLM providers — designed to be the inference layer for the `actdecor` package.

## Supported providers

| Provider | Class | Default model | API key env var |
| --- | --- | --- | --- |
| Moonshot (Kimi) | `ChatMoonshot` / `ChatKimi` | `moonshot-v1-8k` | `MOONSHOT_API_KEY` |
| ZhipuAI (GLM) | `ChatZhipuAI` / `ChatGLM` | `glm-4-plus` | `ZHIPUAI_API_KEY` |
| MiniMax | `ChatMiniMax` | `abab6.5s-chat` | `MINIMAX_API_KEY` |
| DeepSeek | `ChatDeepSeek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| OpenAI | `ChatOpenAI` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `ChatAnthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Hugging Face | `ChatHuggingFace` | `Hythcliff/canadian-address-checker-on` | `HF_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN`) |
## Install

```bash
pip install -e .
```

The only required dependency is `requests`. `httpx` is optional (for users who want async transports later).

## Quick start

### Direct invocation

```python
from actllminfer import ChatKimi, HumanMessage, SystemMessage

llm = ChatKimi(model="moonshot-v1-8k", temperature=0.2)
reply = llm.invoke([
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="Summarize the theory of relativity in one sentence."),
])
print(reply.content)
```

### String spec via the factory

```python
from actllminfer import init_chat_model

llm = init_chat_model("kimi:moonshot-v1-8k", temperature=0)
llm = init_chat_model("glm-4-plus")                 # provider inferred
llm = init_chat_model("deepseek-reasoner")          # provider inferred
llm = init_chat_model("anthropic:claude-sonnet-4-6")
llm = init_chat_model("abab6.5s-chat")              # MiniMax inferred
llm = init_chat_model("hf:meta-llama/Llama-3.3-70B-Instruct")
```
### Hugging Face

`ChatHuggingFace` defaults to the **HF Inference Router**
(`https://router.huggingface.co/v1/chat/completions`), which is
OpenAI-compatible and dispatches to whichever provider currently serves the
model id you pass.

```python
from actllminfer import ChatHuggingFace

llm = ChatHuggingFace(model="Qwen/Qwen2.5-72B-Instruct")
print(llm.invoke("Summarize the theory of relativity in one sentence.").content)
```

For a **dedicated Inference Endpoint**, a self-hosted **TGI** server, or any
other OpenAI-compatible deployment, just point `base_url` at the `/v1` root:

```python
llm = ChatHuggingFace(
    model="tgi",  # placeholder; the endpoint already targets a single model
    base_url="https://my-endpoint.example.com/v1",
)
```

The class accepts `HF_TOKEN` *or* the older `HUGGINGFACEHUB_API_TOKEN` env var.

A worked example using `Hythcliff/canadian-address-checker-on` to validate a
batch of Canadian addresses and parse a structured JSON response is in
[`examples/canadian_address_checker.py`](examples/canadian_address_checker.py).

### Composable chains (LCEL-style)

```python
from actllminfer import ChatPromptTemplate, StrOutputParser, init_chat_model

llm = init_chat_model("kimi:moonshot-v1-8k")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful translator."),
    ("user", "Translate to {language}: {text}"),
])
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"language": "French", "text": "Good morning"}))
```

### Streaming

```python
for chunk in llm.stream("Tell me a short story about a robot."):
    print(chunk.text, end="", flush=True)
```

### Tool / function calling (OpenAI-shaped)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

llm_with_tools = init_chat_model("kimi:moonshot-v1-8k").with_tools(tools)
ai_msg = llm_with_tools.invoke("What's the weather in Beijing?")
for call in ai_msg.tool_calls:
    print(call.name, call.args)
```

The same `tools=[{"type": "function", ...}]` spec is automatically translated to Anthropic's `input_schema` shape for `ChatAnthropic`.

### JSON output

```python
from actllminfer import JsonOutputParser

chain = prompt | llm | JsonOutputParser()
data = chain.invoke({"language": "JSON", "text": "Return {\"ok\": true} only."})
```

## Architecture

```
actllminfer/
├── messages.py         # BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage, ...
├── outputs.py          # ChatGeneration, ChatResult, ChatGenerationChunk
├── prompts.py          # PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
├── output_parsers.py   # StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
├── runnables.py        # Runnable, RunnableLambda, RunnablePassthrough, RunnableSequence
├── callbacks.py        # BaseCallbackHandler, CallbackManager, StdOutCallbackHandler
├── language_models/
│   └── base.py         # BaseChatModel
├── chat_models/
│   ├── _openai_compat.py   # shared OpenAI /v1/chat/completions backend
│   ├── openai.py
│   ├── moonshot.py     # Kimi
│   ├── deepseek.py
│   ├── zhipuai.py      # GLM
│   ├── minimax.py
|   ├── anthropic.py    # Claude (different transport)
│   └── huggingface.py  # HF Inference Router / TGI / dedicated endpoints
├── factory.py          # init_chat_model("kimi:moonshot-v1-8k")
└── exceptions.py
```

Every provider implements the same `BaseChatModel` contract:
`invoke`, `batch`, `stream`, `generate`, `with_tools`, `bind`. That means the `actdecor` package can keep one code path and switch providers via configuration.

## License

Apache 2.0.

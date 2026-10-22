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
| NVIDIA NIM | `ChatNVIDIA` | `meta/llama-3.3-70b-instruct` | `NVIDIA_API_KEY` (or `NGC_API_KEY`) |
## Model prices

`actllminfer.pricing` holds each provider's official list price per million
tokens, for the providers above plus reference prices for Google, xAI,
Mistral, Cohere and Alibaba (which ActLLMInfer has no client for yet). It is
the price source for routers built on ActLLMInfer, such as ActRouter.

```python
from actllminfer.pricing import get_price, estimate_cost

get_price("anthropic/claude-sonnet-4-6")      # ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0, ...)
get_price("glm-4.6").provider                 # "zhipuai" (inferred, like init_chat_model)
estimate_cost("openai/gpt-5-mini", prompt_tokens=1200, completion_tokens=300)  # 0.0009
```

The table is regenerated from LiteLLM's price data, read at each provider's
own route (never a reseller's). Models that disappear upstream or pass their
deprecation date are dropped, and new ones are added:

```bash
python scripts/update_prices.py            # rewrite actllminfer/data/model_prices.json
python scripts/update_prices.py --check    # exit 1 if the bundled table is out of date
```

NVIDIA NIM and Hugging Face publish no per-token list prices, so their
models have no entry and `get_price` returns `None`.

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

### Unified `completion()` endpoint (LiteLLM-style)

For callers that just want a single function with an OpenAI-shaped response,
`completion()` dispatches across every supported provider via a
`provider/model` string. Providers can also be addressed with the legacy
`provider:model` separator, and known prefixes (`glm-`, `deepseek-`,
`claude-`, `nemotron`, …) are inferred automatically.

```python
from actllminfer import completion

resp = completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize relativity in one line."}],
    temperature=0.2,
)
print(resp.choices[0].message.content)   # attribute access
print(resp["choices"][0]["message"]["content"])  # dict access
print(resp.usage["total_tokens"])
```

`stream=True` returns an iterator of OpenAI-shaped `chat.completion.chunk`
objects:

```python
for chunk in completion(model="kimi/moonshot-v1-8k", messages="Tell me a joke", stream=True):
    delta = chunk.choices[0].delta
    if delta.get("content"):
        print(delta.content, end="", flush=True)
```

Per-call generation knobs (`temperature`, `max_tokens`, `tools`,
`response_format`, `seed`, `stop`, …) are forwarded to the provider.
Constructor-time arguments (`api_key`, `base_url`, `organization`,
`request_timeout`, `default_headers`, `extra_body`, …) build (and cache)
the underlying client.

An `embedding()` counterpart and `async` variants (`acompletion`,
`aembedding`) are provided too:

```python
from actllminfer import embedding, acompletion

vecs = embedding(model="openai/text-embedding-3-small", input=["hi", "there"])
print(vecs.data[0].embedding[:4])

resp = await acompletion(model="anthropic/claude-sonnet-4-6", messages="Hi")
```

A primary-with-fallbacks `Router` retries on transient errors so a
rate-limited primary transparently falls through to the next provider:

```python
from actllminfer import Router

router = Router([
    "openai/gpt-4o-mini",
    "kimi/moonshot-v1-8k",
    "deepseek/deepseek-chat",
])
resp = router.completion(messages=[{"role": "user", "content": "Hi"}])
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
llm = init_chat_model("nvidia:meta/llama-3.3-70b-instruct")
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

### NVIDIA NIM (free serverless inference)

`ChatNVIDIA` targets NVIDIA's free OpenAI-compatible NIM endpoint at
`https://integrate.api.nvidia.com/v1/chat/completions`. Grab a free
`nvapi-...` key from [build.nvidia.com](https://build.nvidia.com/) and set
`NVIDIA_API_KEY` (the legacy `NGC_API_KEY` is also accepted).

```python
from actllminfer import ChatNVIDIA

llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2)
print(llm.invoke("Summarize the theory of relativity in one sentence.").content)
```

The same key fans out to dozens of hosted models — Llama 3.x, Mixtral,
Nemotron, Qwen, DeepSeek, Phi, Gemma, etc. To target a self-hosted NIM
microservice instead, point `base_url` at any OpenAI-compatible `/v1` root:

```python
llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    base_url="https://my-nim.example.com/v1",
)
```

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
│   ├── huggingface.py  # HF Inference Router / TGI / dedicated endpoints
│   └── nvidia.py       # NVIDIA NIM serverless / self-hosted
├── factory.py          # init_chat_model("kimi:moonshot-v1-8k")
└── exceptions.py
```

Every provider implements the same `BaseChatModel` contract:
`invoke`, `batch`, `stream`, `generate`, `with_tools`, `bind`. That means the `actdecor` package can keep one code path and switch providers via configuration.

## License

Apache 2.0.

"""ActLLMInfer — integrated LLM inference engine with a LangChain-Core-style API.

Quick start::

    from actllminfer import init_chat_model, ChatPromptTemplate, StrOutputParser

    llm = init_chat_model("kimi:moonshot-v1-8k", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "Translate to {language}: {text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({"language": "French", "text": "Good morning"}))
"""

from .callbacks import BaseCallbackHandler, CallbackManager, StdOutCallbackHandler
from .chat_models import (
    ChatAnthropic,
    ChatDeepSeek,
    ChatGLM,
    ChatHuggingFace,
    ChatKimi,
    ChatMiniMax,
    ChatMoonshot,
    ChatNVIDIA,
    ChatOpenAI,
    ChatZhipuAI,
)
from .exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    ConfigurationError,
    LLMError,
    OutputParserError,
    RateLimitError,
)
from .factory import init_chat_model
from .language_models import BaseChatModel
from .messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    coerce_message,
    coerce_messages,
)
from .output_parsers import (
    BaseOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    StrOutputParser,
)
from .outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from .prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from .runnables import Runnable, RunnableLambda, RunnablePassthrough, RunnableSequence
from .utils import SecretStr

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # factory
    "init_chat_model",
    # base
    "BaseChatModel",
    # provider classes
    "ChatOpenAI",
    "ChatMoonshot",
    "ChatKimi",
    "ChatDeepSeek",
    "ChatZhipuAI",
    "ChatGLM",
    "ChatMiniMax",
    "ChatAnthropic",
    "ChatHuggingFace",
    "ChatNVIDIA",
    # messages
    "BaseMessage",
    "SystemMessage",
    "HumanMessage",
    "AIMessage",
    "AIMessageChunk",
    "ToolMessage",
    "FunctionMessage",
    "ToolCall",
    "coerce_message",
    "coerce_messages",
    # outputs
    "ChatGeneration",
    "ChatGenerationChunk",
    "ChatResult",
    # prompts
    "PromptTemplate",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    # output parsers
    "BaseOutputParser",
    "StrOutputParser",
    "JsonOutputParser",
    "CommaSeparatedListOutputParser",
    # runnables
    "Runnable",
    "RunnableLambda",
    "RunnablePassthrough",
    "RunnableSequence",
    # callbacks
    "BaseCallbackHandler",
    "CallbackManager",
    "StdOutCallbackHandler",
    # utils / exceptions
    "SecretStr",
    "LLMError",
    "ConfigurationError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
    "APIConnectionError",
    "OutputParserError",
]

from actllminfer import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    coerce_message,
    coerce_messages,
)


def test_basic_message_dicts():
    assert SystemMessage(content="hi").to_dict() == {"role": "system", "content": "hi"}
    assert HumanMessage(content="hi").to_dict() == {"role": "user", "content": "hi"}
    assert AIMessage(content="ok").to_dict() == {"role": "assistant", "content": "ok"}


def test_tool_message_carries_call_id():
    d = ToolMessage(content="42", tool_call_id="call_1").to_dict()
    assert d == {"role": "tool", "content": "42", "tool_call_id": "call_1"}


def test_ai_message_with_tool_calls_serializes_function_payload():
    ai = AIMessage(
        content="",
        tool_calls=[ToolCall(id="call_1", name="get_weather", args={"city": "Beijing"})],
    )
    d = ai.to_dict()
    assert d["role"] == "assistant"
    assert d["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "Beijing" in d["tool_calls"][0]["function"]["arguments"]


def test_coerce_str_and_tuple_and_dict():
    assert isinstance(coerce_message("hello"), HumanMessage)
    assert isinstance(coerce_message(("system", "be helpful")), SystemMessage)
    msg = coerce_message({"role": "assistant", "content": "ok"})
    assert isinstance(msg, AIMessage) and msg.content == "ok"


def test_coerce_messages_accepts_mixed_iterable():
    msgs = coerce_messages([
        ("system", "be helpful"),
        "what is 2+2?",
        AIMessage(content="4"),
    ])
    assert [type(m).__name__ for m in msgs] == ["SystemMessage", "HumanMessage", "AIMessage"]


def test_aimessage_chunk_concatenation():
    a = AIMessageChunk(content="Hel")
    b = AIMessageChunk(content="lo")
    merged = a + b
    assert merged.content == "Hello"


def test_aimessage_chunk_merges_tool_calls_by_id():
    a = AIMessageChunk(content="", tool_calls=[ToolCall(id="x", name="f", args={"a": 1})])
    b = AIMessageChunk(content="", tool_calls=[ToolCall(id="x", name="", args={"b": 2})])
    merged = a + b
    assert len(merged.tool_calls) == 1
    assert merged.tool_calls[0].args == {"a": 1, "b": 2}

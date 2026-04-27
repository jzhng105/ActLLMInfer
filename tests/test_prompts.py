import pytest

from actllminfer import (
    AIMessage,
    ChatPromptTemplate,
    HumanMessage,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessage,
)


def test_prompt_template_extracts_variables():
    t = PromptTemplate.from_template("Hello {name}, you are {age} years old")
    assert t.input_variables == ["name", "age"]
    assert t.format(name="Mei", age=12) == "Hello Mei, you are 12 years old"


def test_chat_prompt_template_basic_format():
    p = ChatPromptTemplate.from_messages([
        ("system", "Answer in {language}."),
        ("user", "What is {x} + {y}?"),
    ])
    assert p.input_variables == ["language", "x", "y"]
    out = p.format_messages(language="French", x=1, y=2)
    assert isinstance(out[0], SystemMessage) and out[0].content == "Answer in French."
    assert isinstance(out[1], HumanMessage) and out[1].content == "What is 1 + 2?"


def test_messages_placeholder_splices_history():
    p = ChatPromptTemplate.from_messages([
        ("system", "be helpful"),
        MessagesPlaceholder("history"),
        ("user", "{q}"),
    ])
    history = [HumanMessage(content="hi"), AIMessage(content="hello")]
    out = p.format_messages(history=history, q="how are you?")
    assert [m.content for m in out] == ["be helpful", "hi", "hello", "how are you?"]


def test_messages_placeholder_optional_can_be_skipped():
    p = ChatPromptTemplate.from_messages([
        ("system", "be helpful"),
        MessagesPlaceholder("history", optional=True),
        ("user", "{q}"),
    ])
    out = p.format_messages(q="hi")
    assert len(out) == 2


def test_chat_prompt_invoke_via_dict():
    p = ChatPromptTemplate.from_messages([("user", "Hello {name}")])
    out = p.invoke({"name": "World"})
    assert out[0].content == "Hello World"


def test_missing_required_variable_errors():
    p = ChatPromptTemplate.from_messages([("user", "Hello {name}")])
    with pytest.raises(KeyError):
        p.format_messages()

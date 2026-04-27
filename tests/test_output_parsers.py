import pytest

from actllminfer import (
    AIMessage,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    OutputParserError,
    StrOutputParser,
)


def test_str_parser_passes_text_through():
    assert StrOutputParser().invoke("hello") == "hello"


def test_str_parser_extracts_text_from_message():
    assert StrOutputParser().invoke(AIMessage(content="hi")) == "hi"


def test_json_parser_handles_plain_object():
    assert JsonOutputParser().invoke('{"a": 1}') == {"a": 1}


def test_json_parser_strips_code_fence():
    text = "Here you go:\n```json\n{\"a\": 1, \"b\": [1, 2]}\n```\nHope that helps!"
    assert JsonOutputParser().invoke(text) == {"a": 1, "b": [1, 2]}


def test_json_parser_extracts_object_inside_prose():
    text = "Sure, the answer is {\"ok\": true} as requested."
    assert JsonOutputParser().invoke(text) == {"ok": True}


def test_json_parser_raises_on_garbage():
    with pytest.raises(OutputParserError):
        JsonOutputParser().invoke("definitely not json")


def test_comma_separated_parser_strips_items():
    out = CommaSeparatedListOutputParser().invoke(" red ,green, blue ,")
    assert out == ["red", "green", "blue"]

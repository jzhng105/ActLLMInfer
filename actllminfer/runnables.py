"""Minimal Runnable / LCEL-style composition.

Mirrors the small subset of langchain_core.runnables that the actdecor
package actually needs:

    chain = prompt | model | parser
    chain.invoke({"topic": "bears"})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterator, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")
NewOutput = TypeVar("NewOutput")


class Runnable(ABC, Generic[Input, Output]):
    """Anything that can be ``invoke``‐d, ``batch``‐d, and (optionally) ``stream``‐d."""

    @abstractmethod
    def invoke(self, input: Input, **kwargs: Any) -> Output: ...

    def batch(self, inputs: list[Input], **kwargs: Any) -> list[Output]:
        return [self.invoke(i, **kwargs) for i in inputs]

    def stream(self, input: Input, **kwargs: Any) -> Iterator[Output]:
        # Default: yield the single non-streaming result.
        yield self.invoke(input, **kwargs)

    # ------------------------------------------------------------------ #
    # Composition: ``a | b`` builds a RunnableSequence.
    # ------------------------------------------------------------------ #

    def __or__(self, other: "Runnable[Output, NewOutput] | Callable[[Output], NewOutput]") -> "RunnableSequence":
        return RunnableSequence(self, _ensure_runnable(other))

    def __ror__(self, other: "Callable[[Any], Input]") -> "RunnableSequence":
        return RunnableSequence(_ensure_runnable(other), self)


class RunnableLambda(Runnable[Input, Output]):
    """Wrap a plain function so it participates in the Runnable protocol."""

    def __init__(self, func: Callable[[Input], Output], name: str | None = None) -> None:
        self.func = func
        self.name = name or getattr(func, "__name__", "lambda")

    def invoke(self, input: Input, **kwargs: Any) -> Output:
        return self.func(input)

    def __repr__(self) -> str:
        return f"RunnableLambda({self.name})"


class RunnablePassthrough(Runnable[Any, Any]):
    """Return the input unchanged. Useful as an identity step in a chain."""

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        return input


class RunnableSequence(Runnable[Any, Any]):
    """``a | b | c``: pipe the output of each step into the next."""

    def __init__(self, *steps: Runnable) -> None:
        flat: list[Runnable] = []
        for step in steps:
            if isinstance(step, RunnableSequence):
                flat.extend(step.steps)
            else:
                flat.append(step)
        if not flat:
            raise ValueError("RunnableSequence requires at least one step")
        self.steps: list[Runnable] = flat

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        out = input
        for step in self.steps:
            out = step.invoke(out, **kwargs)
        return out

    def stream(self, input: Any, **kwargs: Any) -> Iterator[Any]:
        # Run the head non-streaming, stream the tail. This is what LangChain
        # does in practice: prompts and parsers don't stream, models do.
        head, *rest = self.steps
        if not rest:
            yield from head.stream(input, **kwargs)
            return
        intermediate = head.invoke(input, **kwargs)
        # Walk through any non-final steps non-streaming.
        for step in rest[:-1]:
            intermediate = step.invoke(intermediate, **kwargs)
        yield from rest[-1].stream(intermediate, **kwargs)

    def __or__(self, other):  # type: ignore[override]
        return RunnableSequence(*self.steps, _ensure_runnable(other))

    def __repr__(self) -> str:
        return " | ".join(repr(s) for s in self.steps)


def _ensure_runnable(obj: Any) -> Runnable:
    if isinstance(obj, Runnable):
        return obj
    if callable(obj):
        return RunnableLambda(obj)
    raise TypeError(f"Cannot compose with {type(obj).__name__}; expected Runnable or callable")


__all__ = [
    "Runnable",
    "RunnableLambda",
    "RunnablePassthrough",
    "RunnableSequence",
]

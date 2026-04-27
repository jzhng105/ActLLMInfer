from actllminfer import RunnableLambda, RunnablePassthrough, RunnableSequence


def test_runnable_lambda_invoke_and_batch():
    r = RunnableLambda(lambda x: x * 2)
    assert r.invoke(3) == 6
    assert r.batch([1, 2, 3]) == [2, 4, 6]


def test_pipe_composes_into_sequence():
    add_one = RunnableLambda(lambda x: x + 1)
    times_two = RunnableLambda(lambda x: x * 2)
    chain = add_one | times_two
    assert isinstance(chain, RunnableSequence)
    assert chain.invoke(3) == 8


def test_three_step_pipeline_flattens():
    a = RunnableLambda(lambda x: x + 1)
    b = RunnableLambda(lambda x: x * 2)
    c = RunnableLambda(lambda x: f"v={x}")
    chain = a | b | c
    assert isinstance(chain, RunnableSequence)
    assert len(chain.steps) == 3
    assert chain.invoke(3) == "v=8"


def test_callable_can_be_piped_in_directly():
    r = RunnableLambda(lambda x: x + 1) | (lambda x: x * 10)
    assert r.invoke(2) == 30


def test_passthrough_returns_input():
    assert RunnablePassthrough().invoke({"k": 1}) == {"k": 1}

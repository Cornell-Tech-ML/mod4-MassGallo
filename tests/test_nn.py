import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors

import random


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # TESTING FORWARD

    out = t.max(dim=2)
    print(out)
    print(t)
    assert_close(out[0, 0, 0], max(t[0, 0, i] for i in range(4)))
    assert_close(out[0, 1, 0], max(t[0, 1, i] for i in range(4)))
    assert_close(out[0, 2, 0], max(t[0, 2, i] for i in range(4)))
    assert_close(out[1, 0, 0], max(t[1, 0, i] for i in range(4)))
    assert_close(out[1, 1, 0], max(t[1, 1, i] for i in range(4)))
    assert_close(out[1, 2, 0], max(t[1, 2, i] for i in range(4)))

    out = t.max(dim=1)
    assert_close(out[0, 0, 0], max(t[0, i, 0] for i in range(3)))
    assert_close(out[0, 0, 1], max(t[0, i, 1] for i in range(3)))
    assert_close(out[0, 0, 2], max(t[0, i, 2] for i in range(3)))
    assert_close(out[0, 0, 3], max(t[0, i, 3] for i in range(3)))
    assert_close(out[1, 0, 0], max(t[1, i, 0] for i in range(3)))
    assert_close(out[1, 0, 1], max(t[1, i, 1] for i in range(3)))
    assert_close(out[1, 0, 2], max(t[1, i, 2] for i in range(3)))
    assert_close(out[1, 0, 3], max(t[1, i, 3] for i in range(3)))

    out = t.max(dim=0)
    assert_close(out[0, 0, 0], max(t[i, 0, 0] for i in range(2)))
    assert_close(out[0, 0, 1], max(t[i, 0, 1] for i in range(2)))
    assert_close(out[0, 0, 2], max(t[i, 0, 2] for i in range(2)))
    assert_close(out[0, 0, 3], max(t[i, 0, 3] for i in range(2)))
    assert_close(out[0, 1, 0], max(t[i, 1, 0] for i in range(2)))
    assert_close(out[0, 1, 1], max(t[i, 1, 1] for i in range(2)))
    assert_close(out[0, 1, 2], max(t[i, 1, 2] for i in range(2)))
    assert_close(out[0, 1, 3], max(t[i, 1, 3] for i in range(2)))

    # TESTING BACKWARD
    noise = minitorch.zeros(t.shape)
    for pos in t._tensor.indices():
        noise[pos] = random.random()

    input = t + noise

    minitorch.grad_check(lambda a: a.max(dim=0), input)
    minitorch.grad_check(lambda a: a.max(dim=1), input)
    minitorch.grad_check(lambda a: a.max(dim=2), input)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)

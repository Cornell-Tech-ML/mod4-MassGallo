"""Collection of the core mathematical operators used throughout the code base.

Disclaimer: AI Claude 3.5 Sonnet (Cursor on Mac) was used to help write comments and code for this file.
"""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the identity of a number."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another."""
    return x < y


def gt(x: float, y: float) -> bool:
    """Checks if one number is greater than another."""
    return x > y


def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid of a number."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU of a number."""
    return x if x > 0 else 0


def log(x: float) -> float:
    """Compute the natural logarithm of a number."""
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential of a number."""
    return float(math.exp(x))


def inv(x: float) -> float:
    """Compute the reciprocal of a number."""
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Compute the derivative of the log function times a second arg."""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Compute the derivative of the reciprocal function times a second arg."""
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Compute the derivative of the ReLU function times a second arg."""
    return y if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element of a list."""
    return [f(x) for x in ls]


def zipWith(
    f: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Apply a function to pairs of elements from two lists."""
    return [f(x1, x2) for x1, x2 in zip(ls1, ls2)]


def reduce(
    f: Callable[[float, float], float], ls: Iterable[float], initial: float
) -> float:
    """Reduce a list to a single value using a function."""
    curr: float = initial
    for x in ls:
        curr = f(curr, x)
    return curr


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two lists."""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Compute the sum of a list of numbers."""
    return reduce(add, ls, 0)


def prod(ls: Iterable[float]) -> float:
    """Compute the product of a list of numbers."""
    return reduce(mul, ls, 1)

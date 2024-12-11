"""Disclaimer: AI Claude 3.5 Sonnet (Cursor on Mac) was used to help write comments and code for this file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus_h = list(vals)
    vals_plus_h[arg] += epsilon
    vals_minus_h = list(vals)
    vals_minus_h[arg] -= epsilon

    f_x_plus_h = f(*vals_plus_h)
    f_x_minus_h = f(*vals_minus_h)

    return (f_x_plus_h - f_x_minus_h) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative of the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Get the unique identifier of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent variables of the current variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Chain rule for backpropagation."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def dfs(node: Variable) -> None:
        """Depth-first search to traverse the computation graph."""
        if node.unique_id in visited or node.is_constant():
            return
        if not node.is_leaf():
            for i in node.parents:
                if not i.is_constant():
                    dfs(i)
        visited.add(node.unique_id)
        order.insert(0, node)

    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:  # noqa: D417
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable : The right-most variable.
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # A dict to store the derivatives for each variable
    # using unique id
    derivatives = {variable.unique_id: deriv}

    # Get variables in topological order
    topo_order = list(topological_sort(variable))

    for node in topo_order:
        d_output = derivatives[node.unique_id]
        # Get the chain rule for the current node
        if node.is_leaf():
            node.accumulate_derivative(d_output)
        else:
            for parent, d_input in node.chain_rule(d_output):
                if parent.is_constant():
                    continue
                derivatives.setdefault(parent.unique_id, 0.0)
                derivatives[parent.unique_id] += d_input


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors from the forward pass."""
        return self.saved_values

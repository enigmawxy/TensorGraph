import numpy as np


class Operation:
    """Represents a graph node that performs a computation.

    An `Operation` is a node in a `Graph` that takes zero or
    more objects as input, and produces zero or more objects
    as output.
    """

    def __init__(self, input_nodes=None):
        """Construct Operation
        """
        if input_nodes is None:
            input_nodes = []
        self.input_nodes = input_nodes

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # Append this operation to the list of operations in the currently active default graph
        DEFAULT_GRAPH.operations.append(self)

    def compute(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass


class add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        return x_value + y_value


class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        """Construct matmul

        Args:
          a: First matrix
          b: Second matrix
        """
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        """Compute the output of the matmul operation

        Args:
          a_value: First matrix value
          b_value: Second matrix value
        """
        return a_value.dot(b_value)


class sigmoid(Operation):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, a):
        """Construct sigmoid

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the sigmoid operation

        Args:
          a_value: Input value
        """
        return 1 / (1 + np.exp(-a_value))


class softmax(Operation):
    """Returns the softmax of a.
    """

    def __init__(self, a):
        """Construct softmax

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the softmax operation

        Args:
          a_value: Input value
        """
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]


class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """

    def __init__(self, x):
        """Construct log

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the log operation

        Args:
          x_value: Input value
        """
        return np.log(x_value)


class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        """Construct multiply

        Args:
          x: First multiplicand node
          y: Second multiplicand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the multiply operation

        Args:
          x_value: First multiplicand value
          y_value: Second multiplicand value
        """

        return x_value * y_value


class reduce_sum(Operation):
    """Computes the sum of elements across dimensions of a tensor.
    """

    def __init__(self, A, axis=None):
        """Construct reduce_sum

        Args:
          A: The tensor to reduce.
          axis: The dimensions to reduce. If `None` (the default), reduces all dimensions.
        """
        super().__init__([A])
        self.axis = axis

    def compute(self, A_value):
        """Compute the output of the reduce_sum operation

        Args:
          A_value: Input tensor value
        """
        return np.sum(A_value, self.axis)


class negative(Operation):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        """Construct negative

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the negative operation

        Args:
          x_value: Input value
        """
        return -x_value


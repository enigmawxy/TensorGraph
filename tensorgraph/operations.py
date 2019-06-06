import numpy as np
from tensorgraph.graph import Operation


class add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y, name=None):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        super(self.__class__, self).__init__(x, y, name=name)

    def compute(self):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        x_value, y_value = self.input_nodes

        self.output_value = np.add(x_value.output_value, y_value.output_value)

        return self.output_value


class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b, name=None):
        """Construct matmul

        Args:
          a: First matrix
          b: Second matrix
        """
        super(self.__class__, self).__init__(a, b, name=name)

    def compute(self):
        """Compute the output of the matmul operation

        Args:
          a_value: First matrix value
          b_value: Second matrix value
        """
        a_value, b_value = self.input_nodes

        self.output_value = np.dot(a_value.output_value, b_value.output_value)

        return self.output_value


class softmax(Operation):
    """Returns the softmax of a.
    """

    def __init__(self, a, name=None):
        """Construct softmax

        Args:
          a: Input node
        """
        super(self.__class__, self).__init__(a, name=name)

    def compute(self):
        """Compute the output of the softmax operation

        Args:
          a_value: Input value
        """
        a = self.input_nodes

        self.output_value = np.exp(a[0].output_value) / np.sum(np.exp(a[0].output_value), axis=1)[:, None]

        return self.output_value


class sigmoid(Operation):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, a, name=None):
        """Construct sigmoid

        Args:
          a: Input node
        """
        super().__init__([a], name)

    def compute(self, a_value):
        """Compute the output of the sigmoid operation

        Args:
          a_value: Input value
        """
        return 1 / (1 + np.exp(-a_value))


class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """

    def __init__(self, x, name=None):
        """Construct log

        Args:
          x: Input node
        """
        super().__init__([x], name)

    def compute(self, x_value):
        """Compute the output of the log operation

        Args:
          x_value: Input value
        """
        return np.log(x_value)


class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y, name=None):
        """Construct multiply

        Args:
          x: First multiplicand node
          y: Second multiplicand node
        """
        super().__init__([x, y], name)

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

    def __init__(self, A, axis=None, name=None):
        """Construct reduce_sum

        Args:
          A: The tensor to reduce.
          axis: The dimensions to reduce. If `None` (the default), reduces all dimensions.
        """
        super().__init__([A], name)
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

    def __init__(self, x, name=None):
        """Construct negative

        Args:
          x: Input node
        """
        super().__init__([x], name)

    def compute(self, x_value):
        """Compute the output of the negative operation

        Args:
          x_value: Input value
        """
        return -x_value


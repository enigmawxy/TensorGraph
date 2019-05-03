import numpy as np
from tensorgraph.graph import Operation
from tensorgraph.common.util import im2col


class con2d(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, x, w, b, strides=None, padding='SAME', name=None):
        super().__init__([x, w, b], name)
        if strides is None:
            self.strides = [1, 1, 1, 1]
        else:
            self.strides = strides
        self.pad = padding
        self.name = name

    def compute(self, x_value, w_value, b_value):
        """Compute the output of the matmul operation

        Args:
          :param x_value: First matrix value
          :param w_value: Second matrix value
          :param b_value:

        """

        FN, C, FH, FW = w_value.shape
        N, C, H, W = x_value.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.strides)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.strides)

        col = im2col(x_value, FH, FW, self.strides, self.pad)
        col_W = w_value.reshape(FN, -1).T

        out = np.dot(col, col_W) + b_value
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


class max_pool(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
       """

    def __init__(self, x, ksize, strides=None, padding='SAME', name=None):
        super().__init__([x, ksize], name)
        if strides is None:
            self.strides = [1, 1, 1, 1]
        else:
            self.strides = strides
        self.pad = padding
        self.name = name

    def compute(self, x_value, ksize_value):
        """Compute the output of the matmul operation

        Args:
          :param x_value: First matrix value
          :param ksize_value: Second matrix value
        """
        N, C, H, W = x_value.shape
        out_h = int(1 + (H - ksize_value[1]) / self.strides)
        out_w = int(1 + (W - ksize_value[2]) / self.strides)

        col = im2col(x_value, ksize_value[1], ksize_value[2], self.strides, self.pad)
        col = col.reshape(-1, ksize_value[1] * ksize_value[2])

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
import numpy as np
from tensorgraph.graph import Operation


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


class con2d(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, x, w, b, strides=None, padding=0, name=None):
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
        out_h = 1 + int((H + 2 * self.pad - FH) / self.strides[1])
        out_w = 1 + int((W + 2 * self.pad - FW) / self.strides[2])

        col = im2col(x_value, FH, FW, self.strides[0], self.pad)
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

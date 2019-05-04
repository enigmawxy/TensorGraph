import numpy as np
from tensorgraph.graph import Operation


def im2col(input_data, filter_h, filter_w, stride=1, pad=None):
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 卷积核的高
    filter_w : 卷积核的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 返回按照(filter_h, filter_w)拉平了数组
    """
    if pad is None:
        pad = [0, 0, 0, 0]

    N, C, H, W = input_data.shape
    out_h = (H + pad[2] + pad[3] - filter_h) // stride + 1
    out_w = (W + pad[0] + pad[1] - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad[0], pad[1]), (pad[2], pad[3])], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


class con2d(Operation):
    def __init__(self, x, w, b, strides=None, padding='VALID', name=None):
        """

        :param x: shape is (N C H W)
        :param w: shape is (F C H W)
        :param b: shape is (W)
        :param strides: is (N C H W)
        :param padding: SAME or VALID
        :param name: node's name
        """
        super().__init__([x, w, b], name)
        if strides is None:
            self.strides = [1, 1, 1, 1]
        else:
            self.strides = strides
        self.padding = padding
        self.name = name

    def compute(self, x_value, w_value, b_value):
        """Compute the output of the matmul operation
            shape: (N C H W)
        Args:
          :param x_value: input data
          :param w_value: filter
          :param b_value: bias

        """

        FN, C, FH, FW = w_value.shape
        N, C, H, W = x_value.shape

        if self.padding is 'SAME':
            if H % self.strides[2] == 0:
                pad_along_height = max(FH - self.strides[2], 0)
            else:
                pad_along_height = max(FH - (H % self.strides[2]), 0)
            if W % self.strides[3] == 0:
                pad_along_width = max(FW - self.strides[3], 0)
            else:
                pad_along_width = max(FW - (W % self.strides[3]), 0)

            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            print(self.strides)
            print(pad_top, pad_bottom, pad_left, pad_right)

            out_h = 1 + int((H + pad_top + pad_bottom - FH) / self.strides[2])
            out_w = 1 + int((W + pad_left + pad_right - FW) / self.strides[3])

            col = im2col(x_value, FH, FW, self.strides[2], [pad_left, pad_right, pad_top, pad_bottom])
        else:
            out_h = 1 + int((H - FH) / self.strides[2])
            out_w = 1 + int((W - FW) / self.strides[3])
            col = im2col(x_value, FH, FW, self.strides[2], [0, 0, 0, 0])

        print('col:', col)

        col_w = w_value.reshape(FN, -1).T

        out = np.dot(col, col_w) + b_value
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

        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

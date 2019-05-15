import numpy as np
import tensorgraph as tg
from PIL import Image
import time

h = 1080
w = 1728
n = 1


def relu(x):
    return np.maximum(0, x)


# N, H, W, C = x.shape            # 分别是批数，高，宽，通道数
# kh, kw, C, kn = filters.shape   # 分别是卷积核的宽，高，通道数，核数
seqs = np.zeros([n] + [h, w] + [3])

w = 0.01 * np.random.randn(3, 3, 3, 1)
b = np.zeros(1)

for i in range(0, n):
    im = Image.open('./images/A0{}.png'.format(i+1))
    seqs[i, :, :] = np.array(im)

print(seqs.shape, w.shape)
start = time.time()
con1 = tg.nn.conv_forward_tensordot(seqs, w, b, 1)
con = relu(con1)


print("执行时间 {} 秒".format(round(time.time() - start, 2)))
print(con.shape)

img = Image.fromarray(con[0], mode='RGB')
img.show()
# split_img = tg.nn.split_by_strides(seqs, 3, 3, 1)
# print(split_img.shape)

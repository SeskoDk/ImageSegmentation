import cv2
import matplotlib.pyplot as plt
import numpy as np

path = "../docs/cat.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# parameters
TYPE = np.int8
KERNEL_SIZE = 5
PADDING_SIZE = KERNEL_SIZE // 2

# blurring kernel
blurr_k = np.ones(shape=(KERNEL_SIZE, KERNEL_SIZE), dtype=TYPE) / KERNEL_SIZE ** 2
padded_image = np.pad(array=image, pad_width=PADDING_SIZE)
height, width = padded_image.shape

K = 3
P = K // 2

A = np.arange(0, 20).reshape(5, 4)
B = np.pad(A, pad_width=P)
H, W = A.shape
for row_i in range(H):
    for col_j in range(W):
        sub_arr = B[row_i: K + row_i, col_j: K + col_j]
        print(sub_arr)

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

# plt.imshow(image, cmap='gray')
# plt.show()

A = np.arange(1, 21).reshape(4, 5)
print(A)


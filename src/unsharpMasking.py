import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""
https://en.wikipedia.org/wiki/Kernel_(image_processing)
"""


def box_blurring(path: str, kernel_size: int = 3, method: str = "blurring") -> np.ndarray:
    K = kernel_size // 2
    padding = kernel_size // 2

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    padded_image = np.pad(image, pad_width=padding)
    new_image = np.zeros(shape=padded_image.shape)

    height, width = image.shape
    # R = H * F
    for row_i in tqdm(range(height), total=height, desc=method):
        for col_j in range(width):
            i = np.minimum(row_i + K, height + K)
            j = np.minimum(col_j + K, width + K)
            # sub_array of the image
            F_i_j = padded_image[row_i: row_i + kernel_size, col_j: col_j + kernel_size]
            # Kernel coefficient
            H_i_j = 1 / (2 * K + 1) ** 2
            # applying blurring by using local Average
            new_image[i][j] = H_i_j * np.sum(F_i_j)

    return new_image[padding:-padding, padding:-padding]


def image_sharpening(path: str, kernel_size: int = 3, alpha: float = 0.9):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blurred_image = box_blurring(path, kernel_size, method="sharpening")
    details = image - blurred_image
    return image + alpha * details


def comparison():
    path = "../docs/Lenna_(test_image).png"
    kernel_size = 5
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blurred_image = box_blurring(path, kernel_size)
    detail = image - blurred_image
    sharped_image = image_sharpening(path, kernel_size)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    fig.suptitle("Image sharpening", fontsize=16)
    ax[0, 0].imshow(image, cmap="gray")
    ax[0, 0].set_title('original image')
    ax[0, 1].imshow(blurred_image, cmap="gray")
    ax[0, 1].set_title(f'smoothed ({kernel_size}x{kernel_size})')
    ax[1, 0].imshow(detail, cmap="gray")
    ax[1, 0].set_title('Details')
    ax[1, 1].imshow(sharped_image, cmap="gray")
    ax[1, 1].set_title('sharped image')

    plot_path = os.path.dirname(path)
    plot_path = os.path.join(plot_path, "sharpening_result.png")
    plt.savefig(plot_path)
    plt.show()


if __name__ == '__main__':
    comparison()


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class AdaptiveThreshold:
    """
    Calculate the local threshold for each pixel using:
    1. the mean of a square neighborhood centered at the pixel (i, j)
    2. a Gaussian filter
    """

    def __init__(self,
                 image_path: str,
                 blockSize: int = 11,
                 C: int = 2,
                 max_value: int = 255,
                 sigma: float = 2) -> None:

        if blockSize % 2 != 1 and blockSize <= 0:
            raise ValueError("block_size must be an odd positive integer")
        elif sigma < 0:
            raise ValueError("sigma must be an positive")

        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.blockSize = blockSize
        self.C = C
        self.max_value = max_value
        self.sigma = sigma

    def adaptive_thresholdMean(self) -> np.ndarray:
        height, width = self.image.shape
        new_image = np.zeros(shape=self.image.shape, dtype=np.uint8)

        for i in tqdm(range(height), total=height, desc="Average Thresholding"):
            for j in range(width):

                left = np.maximum(0, i - self.blockSize // 2)
                top = np.maximum(0, j - self.blockSize // 2)
                right = np.minimum(height - 1, i + self.blockSize // 2)
                down = np.minimum(width - 1, j + self.blockSize // 2)

                visible_area = self.image[left:right + 1, top:down + 1]
                thresh = np.mean(visible_area) - self.C

                if self.image[i, j] >= thresh:
                    new_image[i, j] = self.max_value

        return new_image


    def adaptive_thresholdGaussian(self) -> np.ndarray:
        # create kernel
        kernel = np.zeros((self.blockSize, self.blockSize))
        center = self.blockSize // 2

        for i in range(self.blockSize):
            for j in range(self.blockSize):
                x = i - center - 1
                y = j - center - 1
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * self.sigma ** 2))

        # to blur the image, we have to average the kernel by the sum of the kernel elements
        kernel = kernel / np.sum(kernel)

        # Create the blurred image
        blurred_image = np.zeros_like(self.image)
        padded_image = np.pad(self.image, ((center, center), (center, center)))

        # convolution with image
        height, width = self.image.shape
        for i in tqdm(range(center, height + center), total=height, desc="Gaussian Thresholding"):
            for j in range(center, width + center):
                patch = padded_image[i - center: i + center + 1, j - center: j + center + 1]
                # R_i_j = H_i_j * F_i_j
                blurred_image[i - center, j - center] = np.sum(patch * kernel)

        blurred_image = blurred_image - self.C

        new_image = np.zeros(shape=self.image.shape, dtype=np.uint8)
        new_image[self.image >= blurred_image] = self.max_value

        return new_image


def main():
    path = "../docs/neo.png"
    gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    thresholder = AdaptiveThreshold(image_path=path)

    thresholding_mean = cv2.adaptiveThreshold(src=gray_image, maxValue=255,
                                              adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                              thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholding_gaussian = cv2.adaptiveThreshold(src=gray_image, maxValue=255,
                                                  adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    fig.suptitle('Comparison of thresholding results', fontsize=16)
    fig.tight_layout()
    ax[0, 0].imshow(thresholding_mean)
    ax[0, 0].set_title('openCV: mean')
    ax[0, 1].imshow(thresholding_gaussian)
    ax[0, 1].set_title('openCV: gaussian')
    ax[1, 0].imshow(thresholder.adaptive_thresholdMean())
    ax[1, 0].set_title('Custom: mean')
    ax[1, 1].imshow(thresholder.adaptive_thresholdGaussian())
    ax[1, 1].set_title('Custom: gaussian')

    plot_path = os.path.dirname(path)
    plot_path = os.path.join(plot_path, "thresholding_result.png")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    main()

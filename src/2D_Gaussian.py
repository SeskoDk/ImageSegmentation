import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


def gaussian(R: int) -> np.ndarray:
    return np.array((1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (R ** 2)))


def plot_2D_Gaussian():
    Bound = 5
    X = np.arange(-Bound, Bound, step=0.25)
    Y = np.arange(-Bound, Bound, step=0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = gaussian(R)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.tight_layout()
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlim(0, 0.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02}')
    plt.show()


if __name__ == "__main__":
    plot_2D_Gaussian()

from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from flax.typing import Array
from matplotlib.figure import Figure


def show_img(img: Array, ax=None, title: Optional[str] = None) -> None:
    """Shows a single image."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def show_img_grid(imgs: List[Array], titles: Iterable[str]) -> Figure:
    """Shows a grid of images."""
    n = int(np.ceil(len(imgs) ** 0.5))
    fig, axs = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        show_img(img, axs[i // n][i % n], title)
    return fig

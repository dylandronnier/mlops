from collections.abc import Iterable
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL.Image import Image


def show_img(img: Image, ax=None, title: Optional[str] = None) -> None:
    """Shows a single image."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def show_img_grid(
    imgs: list[Image],
    titles: Iterable[str],
    images_per_row: int,
    images_per_column: int,
) -> Figure:
    """Shows a grid of images."""
    fig, axs = plt.subplots(
        images_per_row,
        images_per_column,
        figsize=(4 * images_per_row, 4 * images_per_column),
    )
    for i, (img, title) in enumerate(zip(imgs, titles)):
        j = i // images_per_row
        show_img(img, axs, title)
    return fig

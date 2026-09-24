"""Color-wheel visualization for 2D displacement fields."""

import matplotlib.pyplot as plt
import numpy as np


def make_colorwheel() -> np.ndarray:
    """
    Create a color wheel for displacement or optical-flow visualization.

    Returns
    -------
    np.ndarray
        Array of RGB values with shape (n_colors, 3).
    """
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr

    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    col = 0

    # Red -> Yellow
    colorwheel[col : col + ry, 0] = 255
    colorwheel[col : col + ry, 1] = np.floor(
        255 * np.arange(ry) / ry
    )
    col += ry

    # Yellow -> Green
    colorwheel[col : col + yg, 0] = 255 - np.floor(
        255 * np.arange(yg) / yg
    )
    colorwheel[col : col + yg, 1] = 255
    col += yg

    # Green -> Cyan
    colorwheel[col : col + gc, 1] = 255
    colorwheel[col : col + gc, 2] = np.floor(
        255 * np.arange(gc) / gc
    )
    col += gc

    # Cyan -> Blue
    colorwheel[col : col + cb, 1] = 255 - np.floor(
        255 * np.arange(cb) / cb
    )
    colorwheel[col : col + cb, 2] = 255
    col += cb

    # Blue -> Magenta
    colorwheel[col : col + bm, 2] = 255
    colorwheel[col : col + bm, 0] = np.floor(
        255 * np.arange(bm) / bm
    )
    col += bm

    # Magenta -> Red
    colorwheel[col : col + mr, 2] = 255 - np.floor(
        255 * np.arange(mr) / mr
    )
    colorwheel[col : col + mr, 0] = 255

    return colorwheel


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert 2D displacement components into an RGB color image.

    Parameters
    ----------
    u : np.ndarray
        Horizontal displacement component.
    v : np.ndarray
        Vertical displacement component.

    Returns
    -------
    np.ndarray
        RGB image with shape (height, width, 3) and dtype uint8.

    Raises
    ------
    ValueError
        If ``u`` and ``v`` do not have the same shape or are not 2D arrays.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")

    if u.ndim != 2:
        raise ValueError("u and v must be 2D arrays.")

    # Replace invalid values without modifying the caller's input arrays.
    u = np.nan_to_num(u, copy=True)
    v = np.nan_to_num(v, copy=True)

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    radius = np.hypot(u, v)
    angle = np.arctan2(u, v) / np.pi

    # Map angle from [-1, 1] to color-wheel indices.
    fk = (angle + 1) / 2 * (ncols - 1)

    k0 = np.floor(fk).astype(np.intp)
    k1 = (k0 + 1) % ncols
    fraction = fk - k0

    image = np.empty((*u.shape, 3), dtype=np.uint8)

    for channel in range(3):
        values = colorwheel[:, channel]

        col0 = values[k0] / 255.0
        col1 = values[k1] / 255.0
        color = (1 - fraction) * col0 + fraction * col1

        inside_unit_circle = radius <= 1

        # Increase saturation with displacement magnitude.
        color[inside_unit_circle] = (
            1
            - radius[inside_unit_circle]
            * (1 - color[inside_unit_circle])
        )

        # Slightly darken values outside the unit circle.
        color[~inside_unit_circle] *= 0.75

        image[:, :, 2 - channel] = (255 * color).astype(np.uint8)

    return image


def main() -> None:
    """Generate an example color wheel and displacement-field visualization."""
    width = 250

    x = np.arange(-width, width + 1)
    y = np.arange(-width, width + 1)

    dis_x, dis_y = np.meshgrid(x, y)

    dis_x = dis_x.astype(float) * 0.0045
    dis_y = dis_y.astype(float) * 0.0045

    displacement_colors = compute_color(dis_x, dis_y) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Color wheel
    axes[0].imshow(displacement_colors, vmin=0, vmax=1)
    axes[0].set_title("Color wheel")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Colored displacement vectors
    axes[1].quiver(
        x,
        y,
        dis_x,
        dis_y,
        color=displacement_colors.reshape(-1, 3),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
    )
    axes[1].axis("scaled")
    axes[1].axis([x[0], x[-1], y[0], y[-1]])
    axes[1].invert_yaxis()
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Colored displacement")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

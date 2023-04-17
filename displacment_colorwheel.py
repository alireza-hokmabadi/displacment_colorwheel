
'''
Author: Alireza Hokmabadi
Email: a.hokmabadi.ee@gmail.com
Title: Color Wheel and Displacement Visualization
Description: This Python code provides a visualization of a color wheel and displacement vectors using matplotlib.
The color wheel is generated using a custom color mapping function that creates a smooth gradient of colors.
The displacement vectors are plotted as arrows on a grid, with the color of each arrow corresponding to the color
from the color wheel based on its location. This code can be used to visually represent optical flow or any other
displacement-related data in a clear and intuitive manner. It is useful for computer vision, image processing,
and graphics applications. You are free to use and modify this code in your own projects with proper attribution.
Happy coding!
'''

import numpy as np
import matplotlib.pyplot as plt

def make_colorwheel():
    """
    Generates a colorwheel for optical flow visualization.

    Returns:
        np.ndarray: A colorwheel represented as an array of RGB values.
    """
    RY, YG, GC, CB, BM, MR = [15, 6, 4, 11, 13, 6]

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)  # r g b

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col:BM + col, 2] = 255
    colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR + col, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    Compute color image from optical flow vectors u and v.

    Parameters:
        u (np.ndarray): Optical flow vector in x-direction
        v (np.ndarray): Optical flow vector in y-direction

    Returns:
        np.ndarray: Color image with optical flow visualization
    """
    colorwheel = make_colorwheel()

    # Identify NaN values in u and v arrays
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)

    # Replace NaN values with zero in u and v arrays
    u[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.hypot(u, v)
    angle = np.arctan2(u, v) / np.pi
    fk = (angle + 1) / 2 * (ncols - 1)  # -1~1 mapped to 1~ncols
    k0 = fk.astype(np.uint8)            # 1, 2, ..., ncols
    k1 = (k0 + 1) % ncols               # Modulo operation to handle edge case
    f = fk - k0

    img = np.empty((k1.shape[0], k1.shape[1], 3), dtype=np.uint8)  # Initialize as uint8 array for better performance
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75                            # out of range
        img[:, :, 2 - i] = (255 * col).astype(np.uint8)  # Use astype() for type casting

    return img

if __name__ == '__main__':
    #  width of the sample displacement array
    w = 250

    # Generate grid of displacement values for optical flow computation
    x = np.arange(-w, w+1, 1)
    y = np.arange(-w, w+1, 1)

    # Create meshgrid of x and y coordinates
    dis_x, dis_y = np.meshgrid(x, y)

    # Scale x and y coordinates by a factor of 0.0045 and convert to float data type
    dis_x = dis_x.astype(float) * 0.0045
    dis_y = dis_y.astype(float) * 0.0045

    # Call compute_color function
    dis_colorwheel = compute_color(dis_x, dis_y) / 255.0

    # Create a subplot with 1 row and 2 columns, and set the figure size
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Plot the color wheel on the first subplot
    ax = axs[0]
    ax.imshow(dis_colorwheel, vmin=0, vmax=1)
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_text("Color wheel")

    # Plot the displacements using quiver plot on the second subplot
    ax = axs[1]
    ax.quiver(x, y, dis_x, dis_y, color=dis_colorwheel.reshape(-1, 3), angles="xy", scale_units='xy', scale=1.0, width=0.003)
    ax.axis("scaled")
    ax.axis([x[0], x[-1], y[0], y[-1]])
    ax.axis("on")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_text("Colored Displacement")

    plt.show()

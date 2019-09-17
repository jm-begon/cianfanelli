import numpy as np
from skimage.io import imread
from skimage import img_as_float
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def load(fpath):
    img = img_as_float(rgb2gray(imread(fpath)))
    threshold = threshold_otsu(img)
    return img < threshold  # inverse color


def img_to_points(bin_img, n_points=1000, max_depth=1.):
    height = bin_img.shape[0]
    points_2d = np.argwhere(bin_img)
    indices = np.arange(len(points_2d))
    np.random.shuffle(indices)
    points_2d = points_2d[indices[:n_points]]

    verticals = points_2d[:, 0]
    horizontals = points_2d[:, 1]
    depths = np.random.rand(n_points, 1)

    zs = -np.array(verticals) + height
    ys = np.array(horizontals)
    xs = np.array(depths)

    return xs, ys, zs


def plot_scatter3D(xs, ys, zs, elevation=45, azimut=45):
    # elevation=0 and azimut=0 is front-view
    # elevation=0, and azimut=90 is side-view
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elevation, azimut)

    ax.set_title("Azimut: {:.2f}".format(azimut))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.axis("off")

    ax.scatter(xs, ys, zs)
    return ax.get_proj()


def plot_save_close(xs, ys, zs, elevation=0, azimut=0,
                    path="../examples/remove_me.png"):
    plot_scatter3D(xs, ys, zs, elevation, azimut)
    try:
        plt.savefig(path)
    finally:
        plt.close()

if __name__ == '__main__':
    np.random.seed(42)
    bin_img = load("../resources/test.png")
    xs, ys, zs = img_to_points(bin_img, n_points=5000)

    azimuts = np.linspace(0, 360, 100)
    num_len = len(str(len(azimuts)))
    for i, azimut in enumerate(azimuts):
        fname = "../examples/test/frame_{}".format(str(i).zfill(num_len))
        plot_save_close(xs, ys, zs, azimut=azimut-90, path=fname)
        # convert -delay 20 frame_*.png -loop 0 test.gif





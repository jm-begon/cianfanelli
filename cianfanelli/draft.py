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


def noise_within_img(bin_img, n_layers=10, p=.1):
    # TODO generate points in a sparse way
    noise = np.random.rand(n_layers, bin_img.shape[0], bin_img.shape[1])
    noise = noise < p
    points = noise * bin_img
    return np.moveaxis(points, 0, 2)


def white_to_points(img):
    # TODO more efficient
    H, W, D = img.shape
    verticals = []
    horizontals = []
    depths = []
    for row in range(H):
        for col in range(W):
            for cha in range(D):
                if img[row, col, cha] > 0:
                    verticals.append(row)
                    horizontals.append(col)
                    depths.append(cha)

    depth_noise = np.random.uniform(-.5, .5, len(depths))

    zs = -np.array(verticals) + H
    ys = np.array(horizontals)
    xs = np.array(depths) + depth_noise

    return xs, ys, zs


def plot_scatter3D(xs, ys, zs, elevation=45, azimut=45):
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


if __name__ == '__main__':
    bin_img = load("../resources/test.png")
    noise = noise_within_img(bin_img, p=0.001)
    xs, ys, zs = white_to_points(noise)
    # proj = plot_scatter3D(xs, ys, zs, 0, 90)  # Side view
    # proj = plot_scatter3D(xs, ys, zs, 0, 0)  # Front view

    azimuts = np.linspace(0, 360, 100)
    num_len = len(str(len(azimuts)))
    for i, azimut in enumerate(azimuts):
        plot_scatter3D(xs, ys, zs, 0, azimut-90)
        plt.savefig("../examples/test/frame_{}".format(str(i).zfill(num_len)))
        plt.close()





from distance_transform import distance_transform_edt
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def example():
    np.random.seed(0)

    # Number of masks, height and width
    n = 3
    h = 64
    w = 128

    # Binary mask to compute edt from
    masks = np.ones((n, h, w), np.uint8)

    # Colors for Voronoi diagram
    color_palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]

    # Colors of pixels with mask = 0
    colors = np.zeros((n, h, w, 3), np.uint8)

    # Initialize masks with random pixels
    for i in range(n):
        for j in range(100):
            x = np.random.randint(w)
            y = np.random.randint(h)
            masks[i, y, x] = 0
            colors[i, y, x] = color_palette[j % len(color_palette)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    masks = torch.from_numpy(masks).to(device)

    # Compute distance transform
    distances, closest_points = distance_transform_edt(masks)

    for i in range(n):
        mask = masks[i].cpu().numpy()
        distance = distances[i].cpu().numpy()
        closest = closest_points[i].cpu().numpy()

        expected_distance = scipy.ndimage.distance_transform_edt(mask)

        voronoi = colors[i][closest[:, :, 0], closest[:, :, 1]]

        plt.figure(figsize=(12, 8))
        for i, (title, img) in enumerate([
            ("binary mask", mask),
            ("voronoi diagram", voronoi),
            ("distance transform", distance),
            ("scipy for comparison", expected_distance),
        ]):
            plt.subplot(2, 2, 1 + i)
            plt.title(title)
            plt.imshow(img, cmap="gray")
        plt.show()

if __name__ == "__main__":
    example()

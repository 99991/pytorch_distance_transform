![](https://raw.githubusercontent.com/99991/pytorch_distance_transform/main/example.png)

# pytorch_distance_transform

For each pixel with `mask != 0`, find the approximately closest pixel with `mask == 0`, similar to [scipy.ndimage.distance_transform_edt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html).

Based on ["Jump flooding in GPU with applications to Voronoi diagram and distance transform" by Rong et al. (2006)](https://dl.acm.org/doi/10.1145/1111411.1111431).

# Example

```python
import torch
from distance_transform import distance_transform_edt

n, h, w = 10, 64, 128
masks = torch.rand(n, h, w) < 0.99

distances, closest_points = distance_transform_edt(masks)

# distances.shape = (n, h, w)
# closest_points.shape = (n, h, w, 2)

y = closest_points[:, :, :, 0]
x = closest_points[:, :, :, 1]

# Display distance field of first mask
import matplotlib.pyplot as plt
plt.imshow(distances[0], cmap="gray")
plt.show()
```

See [example.py](https://github.com/99991/pytorch_distance_transform/blob/main/example.py) for the code to create the figure above.

# Pros

* Fully vectorized
* Computes not only the distance to the closest point, but also its coordinates
* Only native PyTorch functionality, no compilation required
* [Less than 50 lines of code](https://github.com/99991/pytorch_distance_transform/blob/main/distance_transform.py)
* Supports Euclidean distance

# Cons

* Just an approximation (but usually quite close)
* Not differentiable

Note that OpenCV's [distanceTransform](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042) is probably faster on most hardware and computes an exact solution instead of an approximation. If you can manage OpenCV as a dependency, you should use that instead.

# Dependencies

* The distance transform only requires PyTorch.
* The examples require `numpy`, `scipy` and `matplotlib`.

# Related

* [Jump Flood Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Jump_flooding_algorithm)
* [Kornia's Distance transform (currently uses Manhattan distance instead of Euclidean distance)](https://kornia.readthedocs.io/en/latest/contrib.html#kornia.contrib.DistanceTransform)
* [Linear time implementation of Euclidean distance transform and Voronoi diagrams in C](https://github.com/983/df)
* ["Fast Convolutional Distance Transform" by Karam et al.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8686167) describes how to implement an approximate, differentiable Euclidean distance transform

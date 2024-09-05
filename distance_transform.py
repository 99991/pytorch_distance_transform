import torch

def distance_transform_edt(masks):
    """
    For each pixel with mask != 0, find the approximately closest pixel with mask == 0.
    https://en.wikipedia.org/wiki/Jump_flooding_algorithm

    Based on "Jump flooding in GPU with applications to Voronoi diagram and distance transform"
    by Rong et al. (2006).
    https://dl.acm.org/doi/10.1145/1111411.1111431

    Parameters:
        masks: (n, h, w) tensor with binary masks

    Returns:
        distances: (n, h, w) tensor with approximate distances to the closest pixel with mask == 0
        closest: (n, h, w, 2) tensor with (y, x) coordinates of the closest pixel with mask == 0

    """
    def arange(*args):
        return torch.arange(*args, device=masks.device)

    n, h, w = masks.shape
    points = torch.stack(torch.meshgrid(arange(h), arange(w), indexing="ij"), dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)
    closest = torch.full((n, h, w, 2), fill_value=2 * max(w, h), dtype=points.dtype, device=masks.device)

    # Initialize closest points with self for pixels with mask == 0
    closest[masks == 0] = points[masks == 0]
    distances = torch.norm((points - closest).float(), dim=-1)

    # O(log max(h, w)) iterations
    stride = max(h, w) // 2
    while stride >= 1:
        # Compute (x, y) coordinates of neighboring pixels with given stride
        y, x = torch.meshgrid(arange(h), arange(w), indexing="ij")
        y = y.unsqueeze(-1).unsqueeze(-1) + arange(-1, 2).view(1, 1, 3, 1) * stride
        x = x.unsqueeze(-1).unsqueeze(-1) + arange(-1, 2).view(1, 1, 1, 3) * stride
        x = torch.clamp(x, 0, w - 1)
        y = torch.clamp(y, 0, h - 1)

        # Fetch neighbor points
        neighbors = closest[:, y, x].view(n, h, w, 9, 2)

        # Candidates are neighbor points and the current closest points
        candidates = torch.cat([neighbors, closest.view(n, h, w, 1, 2)], dim=3)
        distances = torch.norm((candidates - points.view(n, h, w, 1, 2)).float(), dim=-1)

        # Find the closest point among candidates
        i_closest = torch.argmin(distances, dim=3)
        closest = candidates[arange(n).view(-1, 1, 1), arange(h).view(1, -1, 1), arange(w).view(1, 1, -1), i_closest]
        distances = distances[arange(n).view(-1, 1, 1), arange(h).view(1, -1, 1), arange(w).view(1, 1, -1), i_closest]

        stride >>= 1
    return distances, closest

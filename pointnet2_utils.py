"""
PointNet++ Core Utilities
------------------------------
Low-level operations that PointNet++'s hierarchical architecture is built
from. These are implemented in plain PyTorch (no custom CUDA kernels), so
they run on both CPU and GPU without any special build step -- at the
point-cloud sizes used in this project (1024-4096 points), this is fast
enough; it would need optimized CUDA ops (as in the original paper's
implementation) to scale to tens of thousands of points efficiently.
"""

import torch


def square_distance(src, dst):
    """
    Pairwise squared Euclidean distance between two point sets.
    src: (B, N, C), dst: (B, M, C) -> (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Gathers points/features at the given indices, batched.
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    returns: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """
    Iterative farthest point sampling: greedily picks npoint centers that
    are spread as far apart as possible, so the downsampled set still
    covers the whole shape rather than clustering in one region.
    xyz: (B, N, 3) -> returns indices (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    For each center point in new_xyz, finds up to nsample neighbors
    within `radius` in xyz. If fewer than nsample points fall within the
    radius, the first found neighbor is repeated to pad -- standard
    behavior in the PointNet++ reference implementation.
    xyz: (B, N, 3) -- full point set
    new_xyz: (B, S, 3) -- query centers
    returns: (B, S, nsample) indices into xyz
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N  # mark out-of-radius points with sentinel N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    One Set Abstraction level's core step: pick npoint centers via FPS,
    group each center's local neighborhood via ball query, and return
    the grouped (relative-coordinate) point sets ready for a shared MLP.

    xyz: (B, N, 3), points: (B, N, D) or None (extra per-point features
    from the previous layer, if any)
    returns: new_xyz (B, npoint, 3), new_points (B, npoint, nsample, 3+D)
    """
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)  # center on local origin

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Used for the final Set Abstraction level: groups the ENTIRE point
    cloud into one group, producing a single global feature -- this is
    the hierarchical-features equivalent of plain PointNet's global
    max-pool, but now built on top of two levels of local aggregation
    rather than raw points directly.
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

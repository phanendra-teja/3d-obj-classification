"""
PointNet++ Segmentation Model
------------------------------------
Upgrade from the plain PointNet model in model.py. The key difference:
plain PointNet extracts a feature for every point independently, then
pools once globally. PointNet++ instead builds a HIERARCHY -- repeatedly
grouping points into local neighborhoods and extracting features at
increasing spatial scale (Set Abstraction), then propagating those
features back down to every original point (Feature Propagation).

This local-neighborhood awareness is specifically what plain PointNet
lacks, and is the direct fix for the Screw-classification weakness
observed in training: Screw's geometry is small and spatially scattered,
which needs local context to distinguish well -- a single global feature
summarizing the WHOLE point cloud (plain PointNet's approach) tends to
wash out that kind of fine local detail.

Architecture (single-scale grouping, 3 levels):
    Input (B, N, 3)
      -> SA1: 1024 -> 512 points, local radius 0.2, features -> 128 dim
      -> SA2: 512  -> 128 points, local radius 0.4, features -> 256 dim
      -> SA3: 128  -> 1 point (global), features -> 1024 dim
      -> FP3: propagate 1 point's features back to 128 points
      -> FP2: propagate 128 points' features back to 512 points
      -> FP1: propagate 512 points' features back to all N original points
      -> per-point classifier head -> (B, N, num_classes)

Radius values assume input point clouds are normalized to a unit sphere
(exactly what prepare_dataset.py's normalize_point_cloud does) -- if you
change that normalization, these radii will need retuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import sample_and_group, sample_and_group_all, square_distance, index_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3), points: (B, N, D) or None
        returns: new_xyz (B, npoint_or_1, 3), new_points (B, npoint_or_1, mlp[-1])
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_points: (B, npoint, nsample, C) -> (B, C, nsample, npoint) for Conv2d
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # max over the nsample (local neighborhood) dimension
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, C)

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Propagates features from the sparser point set (xyz2, points2) up
        to the denser point set (xyz1), via inverse-distance-weighted
        interpolation from each xyz1 point's 3 nearest neighbors in xyz2.
        points1 (skip connection from the matching encoder level, or None
        at the very first level where there are no extra input features)
        is concatenated on before the shared MLP.

        xyz1: (B, N, 3) -- target (denser) points
        xyz2: (B, S, 3) -- source (sparser) points
        points1: (B, N, D1) or None -- skip-connected features at xyz1
        points2: (B, S, D2) -- features at xyz2, to be propagated
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # only one source point (e.g. propagating from the global SA3
            # feature) -- broadcast it to every target point directly
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3 nearest neighbors

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # (B, D, N) for Conv1d
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)  # (B, N, D_out)


class PointNet2Segmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Encoder: progressively downsample, extracting increasingly
        # large-scale local features at each level.
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                           in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                           in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                           in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # Decoder: propagate features back down to the original point count.
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])

        self.classifier_conv1 = nn.Conv1d(128, 128, 1)
        self.classifier_bn1 = nn.BatchNorm1d(128)
        self.classifier_dropout = nn.Dropout(0.5)
        self.classifier_conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """
        xyz: (batch, num_points, 3)
        returns: (batch, num_points, num_classes)
        """
        l0_xyz = xyz
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = l0_points.permute(0, 2, 1)  # (B, 128, N) for Conv1d
        x = F.relu(self.classifier_bn1(self.classifier_conv1(x)))
        x = self.classifier_dropout(x)
        x = self.classifier_conv2(x)  # (B, num_classes, N)

        return x.permute(0, 2, 1)  # (B, N, num_classes)


if __name__ == "__main__":
    model = PointNet2Segmentation(num_classes=3)
    dummy_input = torch.randn(2, 1024, 3)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}  (expected: [2, 1024, 3])")
    assert output.shape == (2, 1024, 3), "Shape mismatch!"
    print("Shape check passed.")

    # Confirm gradients flow end-to-end
    loss = output.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"All parameters received gradients: {has_grad}")

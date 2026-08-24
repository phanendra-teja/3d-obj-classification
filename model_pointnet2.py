"""
PointNet++ Segmentation Model (Support for 3D XYZ + 3D Surface Normals = 6D Input)
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
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # Max pooling over local neighborhood
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
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
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

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)


class PointNet2Segmentation(nn.Module):
    def __init__(self, num_classes, in_channels=6):
        super().__init__()
        self.in_channels = in_channels
        extra_feature_dim = in_channels - 3  # Normal features (3D)

        # Encoder Architecture
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                           in_channel=3 + extra_feature_dim, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                           in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                           in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # Decoder Architecture
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + extra_feature_dim, mlp=[128, 128, 128])

        self.classifier_conv1 = nn.Conv1d(128, 128, 1)
        self.classifier_bn1 = nn.BatchNorm1d(128)
        self.classifier_dropout = nn.Dropout(0.5)
        self.classifier_conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        """
        x: (B, N, in_channels) -> contains XYZ coordinates and optional Normals features
        """
        l0_xyz = x[:, :, :3]
        l0_points = x[:, :, 3:] if self.in_channels > 3 else None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x_out = l0_points.permute(0, 2, 1)
        x_out = F.relu(self.classifier_bn1(self.classifier_conv1(x_out)))
        x_out = self.classifier_dropout(x_out)
        x_out = self.classifier_conv2(x_out)

        return x_out.permute(0, 2, 1)
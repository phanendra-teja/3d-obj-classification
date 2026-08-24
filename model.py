"""
PointNet Segmentation Model
--------------------------------
Standard PointNet architecture (Qi et al., 2017) adapted for per-point
component classification: given a point cloud sampled from a whole phone's
labeled components, predict which component each point belongs to.

Architecture:
  1. Shared per-point MLP extracts local features for every point
     independently (implemented as 1D convolutions with kernel size 1,
     which is mathematically identical to a per-point fully connected
     layer applied to every point).
  2. A global max-pool over all points produces one global feature vector
     summarizing the whole point cloud's shape.
  3. The global feature is concatenated back onto every point's local
     feature -- this is what lets each point's prediction be informed by
     the overall shape context, not just its own local geometry.
  4. A second per-point MLP outputs a class score for every point.

This omits PointNet's T-Net input/feature alignment transforms for
simplicity -- they help with rotation robustness but add complexity and
aren't required to get a working baseline. Can be added later if
rotation invariance becomes a problem (e.g. if your point clouds aren't
consistently oriented).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Shared per-point feature extractor (local features)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Per-point classifier head (local features [64] + global feature [1024] = 1088)
        self.seg_conv1 = nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)

        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        """
        x: (batch, num_points, 3)
        returns: (batch, num_points, num_classes) -- per-point class scores
        """
        x = x.transpose(2, 1)  # -> (batch, 3, num_points) for Conv1d

        x = F.relu(self.bn1(self.conv1(x)))
        point_features = F.relu(self.bn2(self.conv2(x)))  # (batch, 64, N) -- kept for skip connection
        x = F.relu(self.bn3(self.conv3(point_features)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))  # (batch, 1024, N)

        global_feature = torch.max(x, 2, keepdim=True)[0]  # (batch, 1024, 1)
        num_points = point_features.shape[2]
        global_feature_expanded = global_feature.repeat(1, 1, num_points)  # (batch, 1024, N)

        combined = torch.cat([point_features, global_feature_expanded], dim=1)  # (batch, 1088, N)

        x = F.relu(self.seg_bn1(self.seg_conv1(combined)))
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = F.relu(self.seg_bn3(self.seg_conv3(x)))
        x = self.seg_conv4(x)  # (batch, num_classes, N)

        return x.transpose(2, 1)  # (batch, N, num_classes)


if __name__ == "__main__":
    # Quick shape sanity check
    model = PointNetSegmentation(num_classes=3)
    dummy_input = torch.randn(2, 1536, 3)  # batch of 2, 1536 points each, xyz
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}  (expected: [2, 1536, 3])")
    assert output.shape == (2, 1536, 3), "Shape mismatch!"
    print("Shape check passed.")

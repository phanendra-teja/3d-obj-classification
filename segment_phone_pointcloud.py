"""
Phone Component Point Cloud Segmentation & Interactive 3D Visualizer
------------------------------------------------------------------
1. Samples dense point clouds (20,480 points) from whole CAD models.
2. Applies PointNet++ 4-class segmentation (Battery, Camera, Screw, Other).
3. Cleans boundary noise using k-NN Spatial Majority Voting on points.
4. Generates an interactive 3D Plotly HTML viewer and a PLY point cloud.
"""

import argparse
import os
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
import plotly.graph_objects as go

from model import PointNetSegmentation
from model_pointnet2 import PointNet2Segmentation

# High-contrast RGB Colors
CLASS_COLORS = {
    "Battery": [230, 30, 30],      # Bright Red
    "Camera": [30, 220, 30],       # Bright Green
    "Screw": [30, 110, 240],       # Bright Blue
    "Other": [170, 170, 170],      # Grey Chassis
}


def normalize_features_6d(points, normals):
    """Applies exact training normalization (Unit sphere scaling + Unit normals)."""
    centroid = points.mean(axis=0)
    points_norm = points - centroid
    max_dist = np.max(np.linalg.norm(points_norm, axis=1))
    if max_dist > 0:
        points_norm = points_norm / max_dist

    norm_mags = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_mags[norm_mags == 0] = 1.0
    normals_norm = normals / norm_mags

    return np.hstack([points_norm, normals_norm]).astype(np.float32)


def apply_point_knn_smoothing(points, raw_preds, num_classes, k=16):
    """Filters point classification noise by taking a majority vote of local 3D neighbors."""
    tree = cKDTree(points)
    _, neighbor_indices = tree.query(points, k=k)

    smoothed_preds = np.copy(raw_preds)
    for i, neighbors in enumerate(neighbor_indices):
        neighbor_classes = raw_preds[neighbors]
        counts = np.bincount(neighbor_classes, minlength=num_classes)
        smoothed_preds[i] = np.argmax(counts)

    return smoothed_preds


def export_plotly_html(points, preds, class_names, out_html):
    """Generates an interactive 3D WebGL point cloud visualization in HTML."""
    color_strings = []
    for p in preds:
        c_name = class_names[p]
        rgb = CLASS_COLORS.get(c_name, [170, 170, 170])
        color_strings.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2.5,
            color=color_strings,
            opacity=0.95
        )
    )])

    fig.update_layout(
        title="Phone 3D Component Segmentation",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="white"
    )

    fig.write_html(out_html)
    print(f"\n1. Saved interactive 3D Plotly point cloud to: `{out_html}`")


def segment_phone(model_path, mesh_path, num_points, out_prefix, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = [str(c) for c in checkpoint["class_names"]]
    architecture = checkpoint.get("architecture", "pointnet2")
    in_channels = checkpoint.get("in_channels", 6)

    print(f"Loaded checkpoint: {architecture}, {in_channels}D features, classes={class_names}")

    if architecture == "pointnet2":
        model = PointNet2Segmentation(num_classes=len(class_names), in_channels=in_channels).to(device)
    else:
        model = PointNetSegmentation(num_classes=len(class_names)).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print(f"Sampling {num_points} dense points across surface...")
    sampled_points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    sampled_normals = mesh.face_normals[face_idx]

    features = normalize_features_6d(sampled_points, sampled_normals)

    with torch.no_grad():
        input_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        raw_preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    print("Applying spatial k-NN smoothing pass to clean point boundaries...")
    preds = apply_point_knn_smoothing(sampled_points, raw_preds, len(class_names), k=16)

    print("\n" + "=" * 65)
    print(f" 3D POINT CLOUD SEGMENTATION REPORT: {os.path.basename(mesh_path)}")
    print("=" * 65)

    colors = np.zeros((num_points, 3), dtype=np.uint8)

    for i, c_name in enumerate(class_names):
        mask = (preds == i)
        count = int(mask.sum())
        pct = (count / num_points) * 100

        rgb = CLASS_COLORS.get(c_name, [170, 170, 170])
        colors[mask] = rgb

        if c_name == "Other":
            print(f"  * {c_name:<10}: {count} points ({pct:.1f}% surface)")
            continue

        is_detected = count >= 25
        if is_detected:
            mean_conf = probs[mask, i].mean().item() * 100
            pts = sampled_points[mask]
            centroid = pts.mean(axis=0)
            print(f"  * {c_name:<10}: PRESENT  ({count} points | {pct:.1f}% surface | Conf: {mean_conf:.1f}%)")
            print(f"    - Centroid (XYZ): [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
        else:
            print(f"  * {c_name:<10}: NOT DETECTED")

    print("=" * 65)

    # Export 1: Interactive Plotly HTML file
    out_html = f"{out_prefix}_pointcloud_3d.html"
    export_plotly_html(sampled_points, preds, class_names, out_html)

    # Export 2: Standard PLY point cloud file
    out_ply = f"{out_prefix}_pointcloud.ply"
    pcd = trimesh.points.PointCloud(vertices=sampled_points, colors=colors)
    pcd.export(out_ply)
    print(f"2. Saved PLY point cloud file to: `{out_ply}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pointnet2_4class.pt")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--num_points", type=int, default=20480)
    parser.add_argument("--out_prefix", default="phone10")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment_phone(args.model, args.mesh, args.num_points, args.out_prefix, device)
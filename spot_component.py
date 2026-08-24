"""
Spotter Tool: Check presence of a component (e.g., Camera) on a phone 
and generate a clean Blender visual highlight.
"""

import argparse
import os
import numpy as np
import torch
import trimesh

from model import PointNetSegmentation
from model_pointnet2 import PointNet2Segmentation


def normalize_features_6d(points, normals):
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist

    norm_mags = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_mags[norm_mags == 0] = 1.0
    normals = normals / norm_mags

    return np.hstack([points, normals]).astype(np.float32)


def spot_component(model_path, mesh_path, target_component, num_points, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = [str(c) for c in checkpoint["class_names"]]
    architecture = checkpoint.get("architecture", "pointnet2")
    in_channels = checkpoint.get("in_channels", 6)

    if target_component.capitalize() not in class_names:
        print(f"Error: '{target_component}' is not in trained classes: {class_names}")
        return

    target_idx = class_names.index(target_component.capitalize())

    if architecture == "pointnet2":
        model = PointNet2Segmentation(num_classes=len(class_names), in_channels=in_channels).to(device)
    else:
        model = PointNetSegmentation(num_classes=len(class_names)).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mesh = trimesh.load(mesh_path, force="mesh")
    sampled_points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    sampled_normals = mesh.face_normals[face_idx]

    features = normalize_features_6d(sampled_points, sampled_normals)

    with torch.no_grad():
        input_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    target_mask = (preds == target_idx)
    target_count = target_mask.sum()
    pct = (target_count / num_points) * 100

    print("\n" + "=" * 50)
    print(f"COMPONENT SPOTTER REPORT: {target_component.upper()} ON {mesh_path}")
    print("=" * 50)

    if target_count < 10:
        print(f"STATUS: NOT FOUND / NOT DETECTED (Only {target_count} isolated points predicted)")
        print("=" * 50)
        return

    mean_conf = probs[target_mask, target_idx].mean().item()
    target_pts = sampled_points[target_mask]

    print(f"STATUS: FOUND AND SPOTTED!")
    print(f"  - Points Detected: {target_count} / {num_points} ({pct:.1f}% of phone surface)")
    print(f"  - Model Confidence: {mean_conf * 100:.1f}%")
    
    # Compute 3D Bounding Box location of the component
    min_bound, max_bound = target_pts.min(axis=0), target_pts.max(axis=0)
    center = target_pts.mean(axis=0)
    print(f"  - Centroid Location (XYZ): [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

    # Build a visual scene in Blender: Phone mesh in clean white + Bright Red Box highlighting the component
    phone_mesh = mesh.copy()
    phone_mesh.visual.face_colors = [220, 220, 220, 150]  # Semi-transparent light grey phone

    # Create bounding box marker around spotted component
    bbox_extents = max_bound - min_bound + 0.005  # Slight padding
    marker_box = trimesh.creation.box(extents=bbox_extents)
    marker_box.apply_translation(center)
    marker_box.visual.face_colors = [255, 30, 30, 255]  # Bright Solid Red Box

    scene = trimesh.Scene([phone_mesh, marker_box])
    out_glb = f"spotted_{target_component.lower()}.glb"
    scene.export(out_glb)

    print(f"\nVisual highlight exported to: `{out_glb}`")
    print(f"Import `{out_glb}` into Blender to view the red box spotlighting the {target_component}.")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pointnet2_4class.pt")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--component", required=True, help="Battery, Camera, or Screw")
    parser.add_argument("--num_points", type=int, default=16384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spot_component(args.model, args.mesh, args.component, args.num_points, device)
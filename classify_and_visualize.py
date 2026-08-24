"""
Phone Component Classifier & Visualizer with Spatial Smoothing
--------------------------------------------------------------
1. Evaluates PointNet++ 4-class predictions.
2. Applies k-NN Spatial Majority Voting across face neighborhoods to eliminate ragged edge noise.
3. Exports clean interactive HTML and Blender multi-material OBJ.
"""

import argparse
import os
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

CLASS_COLORS = {
    "Battery": [225, 30, 30, 255],     # Red
    "Camera": [30, 225, 30, 255],      # Green
    "Screw": [30, 90, 225, 255],       # Blue
    "Other": [180, 180, 180, 255],     # Grey Chassis
}


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


def apply_spatial_majority_voting(mesh, raw_face_preds, num_classes, k_neighbors=15):
    """Smooths face predictions by taking the majority vote of local neighboring faces."""
    face_centers = mesh.triangles_center
    face_tree = cKDTree(face_centers)
    _, neighbor_indices = face_tree.query(face_centers, k=k_neighbors)

    smoothed_preds = np.copy(raw_face_preds)
    for f_idx, neighbors in enumerate(neighbor_indices):
        neighbor_classes = raw_face_preds[neighbors]
        counts = np.bincount(neighbor_classes, minlength=num_classes)
        smoothed_preds[f_idx] = np.argmax(counts)

    return smoothed_preds


def classify_and_export(model_path, mesh_path, num_points, prefix, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = [str(c) for c in checkpoint["class_names"]]
    architecture = checkpoint.get("architecture", "pointnet2")
    in_channels = checkpoint.get("in_channels", 6)

    from model import PointNetSegmentation
    from model_pointnet2 import PointNet2Segmentation

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

    # Map sampled points to mesh faces
    face_centers = mesh.triangles_center
    tree = cKDTree(sampled_points)
    _, nearest_sampled_idx = tree.query(face_centers)
    raw_face_preds = preds[nearest_sampled_idx]

    # Apply spatial majority voting smoothing
    print("Applying spatial majority voting pass across face neighborhoods...")
    smoothed_face_preds = apply_spatial_majority_voting(mesh, raw_face_preds, len(class_names), k_neighbors=15)

    print("\n" + "=" * 65)
    print(f" PHONE COMPONENT DETECTION REPORT: {os.path.basename(mesh_path)}")
    print("=" * 65)

    for i, c_name in enumerate(class_names):
        if c_name == "Other":
            continue
        mask = (smoothed_face_preds == i)
        count = int(mask.sum())
        if count >= 20:
            centers = face_centers[mask]
            centroid = centers.mean(axis=0)
            print(f"  * {c_name:<10}: PRESENT  ({count} faces | Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}])")
        else:
            print(f"  * {c_name:<10}: NOT DETECTED")

    print("=" * 65)

    # Build smooth sub-meshes with distinct materials
    scene_objects = []
    for i, c_name in enumerate(class_names):
        f_mask = (smoothed_face_preds == i)
        if f_mask.sum() == 0:
            continue

        sub_mesh = mesh.submesh([f_mask], append=True)
        rgba = CLASS_COLORS.get(c_name, [180, 180, 180, 255])
        material = trimesh.visual.material.SimpleMaterial(diffuse=rgba[:3], ambient=rgba[:3])
        sub_mesh.visual.material = material
        scene_objects.append(sub_mesh)

    scene = trimesh.Scene(scene_objects)

    # Export interactive HTML
    html_path = f"{prefix}_smoothed.html"
    with open(html_path, "w") as f:
        f.write(scene.show(embed_in_page=True))
    print(f"\n1. Smoothed 3D Webpage saved to: `{html_path}`")

    # Export OBJ for Blender
    obj_path = f"{prefix}_smoothed_blender.obj"
    scene.export(obj_path)
    print(f"2. Smoothed OBJ saved to: `{obj_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pointnet2_4class.pt")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--num_points", type=int, default=16384)
    parser.add_argument("--out_prefix", default="phone10")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classify_and_export(args.model, args.mesh, args.num_points, args.out_prefix, device)
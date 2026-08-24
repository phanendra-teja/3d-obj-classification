"""
Inference: Predict 4-Class Components on Whole Unsegmented Mesh
"""

import argparse
import glob
import os
import numpy as np
import torch
import trimesh

from model import PointNetSegmentation
from model_pointnet2 import PointNet2Segmentation

CLASS_COLORS = {
    "Battery": [220, 60, 60, 255],   # Red
    "Camera": [60, 200, 90, 255],    # Green
    "Screw": [60, 100, 220, 255],    # Blue
    "Other": [150, 150, 150, 255],   # Grey (Chassis/Frame)
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


def create_colored_spheres(points, preds, class_names, radius=0.008):
    spheres = []
    base_sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)

    for pt, pred in zip(points, preds):
        c_name = class_names[pred]
        color = CLASS_COLORS.get(c_name, [128, 128, 128, 255])

        s = base_sphere.copy()
        s.apply_translation(pt)
        s.visual.face_colors = color
        spheres.append(s)

    return trimesh.util.concatenate(spheres)


def load_mesh_or_combine(target_path):
    if os.path.isdir(target_path):
        obj_files = sorted(glob.glob(os.path.join(target_path, "*.obj")))
        if not obj_files:
            raise FileNotFoundError(f"No .obj files found inside folder `{target_path}`.")
        meshes = [trimesh.load(f, force="mesh") for f in obj_files]
        return trimesh.util.concatenate(meshes)

    if os.path.isfile(target_path):
        return trimesh.load(target_path, force="mesh")

    raise FileNotFoundError(f"Could not find valid file or folder at `{target_path}`.")


def predict(model_path, mesh_path, num_points, out_glb, out_npz, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = [str(c) for c in checkpoint["class_names"]]
    architecture = checkpoint.get("architecture", "pointnet2")
    in_channels = checkpoint.get("in_channels", 6)

    if architecture == "pointnet2":
        model = PointNet2Segmentation(num_classes=len(class_names), in_channels=in_channels).to(device)
    else:
        model = PointNetSegmentation(num_classes=len(class_names)).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mesh = load_mesh_or_combine(mesh_path)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]

    if in_channels == 6:
        features = normalize_features_6d(points, normals)
    else:
        centroid = points.mean(axis=0)
        points_norm = points - centroid
        max_dist = np.max(np.linalg.norm(points_norm, axis=1))
        features = (points_norm / max_dist if max_dist > 0 else points_norm).astype(np.float32)

    with torch.no_grad():
        input_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidences, preds = probs.max(dim=-1)
        preds = preds.squeeze(0).cpu().numpy()
        confidences = confidences.squeeze(0).cpu().numpy()

    print("\nPredicted point distribution across whole phone:")
    for i, name in enumerate(class_names):
        count = int((preds == i).sum())
        print(f"  {name}: {count} points ({100 * count / num_points:.1f}%)")

    colored_mesh = create_colored_spheres(points, preds, class_names, radius=0.008)
    colored_mesh.export(out_glb)
    print(f"Saved visual 3D model to: {out_glb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pointnet2_4class.pt")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--out_glb", default="test_whole_phone_4class.glb")
    parser.add_argument("--out_npz", default="predicted_labels.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(args.model, args.mesh, args.num_points, args.out_glb, args.out_npz, device)
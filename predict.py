"""
Inference: Predict Components & Export Separated CAD Meshes
"""

import argparse
import glob
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


def predict_and_export_separated_meshes(model_path, mesh_path, num_points, output_folder, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = [str(c) for c in checkpoint["class_names"]]
    architecture = checkpoint.get("architecture", "pointnet2")
    in_channels = checkpoint.get("in_channels", 6)

    print(f"Loaded checkpoint: architecture={architecture}, features={in_channels}D, classes={class_names}")

    if architecture == "pointnet2":
        model = PointNet2Segmentation(num_classes=len(class_names), in_channels=in_channels).to(device)
    else:
        model = PointNetSegmentation(num_classes=len(class_names)).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Target CAD Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Sample points for PointNet++
    sampled_points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    sampled_normals = mesh.face_normals[face_idx]

    if in_channels == 6:
        features = normalize_features_6d(sampled_points, sampled_normals)
    else:
        centroid = sampled_points.mean(axis=0)
        norm_pts = sampled_points - centroid
        max_d = np.max(np.linalg.norm(norm_pts, axis=1))
        features = norm_pts / max_d if max_d > 0 else norm_pts

    with torch.no_grad():
        input_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        logits = model(input_tensor)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    # Map predictions back to nearest mesh faces (unpacking 3 return values)
    proximity = trimesh.proximity.ProximityQuery(mesh)
    _, _, face_nearest = proximity.on_surface(sampled_points)
    
    # Assign class label to each face based on nearest sampled prediction
    face_labels = np.full(len(mesh.faces), -1, dtype=int)
    for p_idx, f_idx in enumerate(face_nearest):
        face_labels[f_idx] = preds[p_idx]

    # Fill unmapped faces using Other/Chassis default
    unmapped = np.where(face_labels == -1)[0]
    if len(unmapped) > 0:
        other_idx = class_names.index("Other") if "Other" in class_names else 0
        face_labels[unmapped] = other_idx

    os.makedirs(output_folder, exist_ok=True)
    print(f"\nExporting separated component meshes to folder: `{output_folder}`")

    for i, c_name in enumerate(class_names):
        mask = (face_labels == i)
        count = mask.sum()
        print(f"  {c_name}: {count} faces")

        if count > 0:
            sub_mesh = mesh.submesh([mask], append=True)
            out_file = os.path.join(output_folder, f"Predicted_{c_name}.obj")
            sub_mesh.export(out_file)

    print(f"\nSaved separated OBJ files to `{output_folder}`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pointnet2_4class.pt")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--num_points", type=int, default=16384)
    parser.add_argument("--output_folder", default="predicted_phone10")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_and_export_separated_meshes(args.model, args.mesh, args.num_points, args.output_folder, device)
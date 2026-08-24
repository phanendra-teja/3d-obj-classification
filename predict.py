"""
Inference: Predict Components on a Whole, Unsegmented Mesh
------------------------------------------------------------------
This is the actual payoff of training: given a phone mesh with NO vertex
groups, NO manual Blender labeling -- just a raw obj file -- predict
which points belong to Battery, Camera, or Screw.

Usage:
    python predict.py --model pointnet_seg_model.pt --mesh new_phone.obj \
        --num_points 4096 --out predicted_labels.glb

Output:
    - Console: predicted point count per class
    - A .glb file you can open in Blender: the WHOLE mesh's surface,
      colored by predicted component (one color per class), so you can
      visually check whether the prediction looks sane.
    - A .npz with the raw sampled points + predicted labels, in case you
      want to feed these into the downstream classical pipeline (CoACD /
      collision checking) instead of manually-exported components.

READ THIS: the model only knows Battery/Camera/Screw (see README
limitation #2). Every single point gets FORCED into one of these 3
classes, even points that are actually frame/chassis/unlabeled geometry.
Don't expect clean boundaries on parts of the mesh outside what the
model was trained to recognize -- this is exactly why the colored glb
output matters: LOOK at it before trusting the predictions blindly.
"""

import argparse
import numpy as np
import torch
import trimesh

from model import PointNetSegmentation
from model_pointnet2 import PointNet2Segmentation

CLASS_COLORS = {
    "Battery": [220, 60, 60, 255],
    "Camera": [60, 200, 90, 255],
    "Screw": [60, 100, 220, 255],
}


def normalize_point_cloud(points):
    """Must match the normalization used in prepare_dataset.py exactly."""
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    return points, centroid, max_dist


def predict(model_path, mesh_path, num_points, out_glb, out_npz, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    architecture = checkpoint.get("architecture", "pointnet")  # older checkpoints predate this field
    print(f"Loaded model trained on classes: {class_names} (architecture: {architecture})")

    model = (PointNet2Segmentation(num_classes=len(class_names)) if architecture == "pointnet2"
              else PointNetSegmentation(num_classes=len(class_names))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Loaded {mesh_path}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    points = points.astype(np.float32)
    print(f"Sampled {num_points} points from the surface")

    norm_points, centroid, scale = normalize_point_cloud(points)

    with torch.no_grad():
        input_tensor = torch.from_numpy(norm_points).float().unsqueeze(0).to(device)  # (1, N, 3)
        logits = model(input_tensor)  # (1, N, num_classes)
        probs = torch.softmax(logits, dim=-1)
        confidences, preds = probs.max(dim=-1)
        preds = preds.squeeze(0).cpu().numpy()
        confidences = confidences.squeeze(0).cpu().numpy()

    print("\nPredicted point counts per class:")
    for i, name in enumerate(class_names):
        count = int((preds == i).sum())
        print(f"  {name}: {count} points ({100 * count / num_points:.1f}%)")
    print(f"\nMean prediction confidence: {confidences.mean():.3f} "
          f"(low confidence suggests the model is unsure -- often means "
          f"the true region isn't one of its {len(class_names)} known classes)")

    # Colored point cloud for visual inspection
    colors = np.array([CLASS_COLORS.get(class_names[p], [128, 128, 128, 255]) for p in preds])
    point_cloud = trimesh.points.PointCloud(points, colors=colors)
    scene = trimesh.Scene([point_cloud])
    scene.export(out_glb)
    print(f"\nSaved colored prediction to {out_glb} -- open in Blender to visually check the result.")
    print("Color legend: " + ", ".join(f"{name}={CLASS_COLORS[name][:3]}" for name in class_names))

    np.savez(
        out_npz,
        points=points,
        predicted_labels=preds,
        confidences=confidences,
        class_names=np.array(class_names),
    )
    print(f"Saved raw predictions to {out_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--mesh", required=True, help="Path to a whole, unsegmented phone obj file")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--out_glb", default="predicted_labels.glb")
    parser.add_argument("--out_npz", default="predicted_labels.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(args.model, args.mesh, args.num_points, args.out_glb, args.out_npz, device)
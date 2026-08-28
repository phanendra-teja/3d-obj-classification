"""
Run the trained model on a NEW, unlabeled phone mesh and flag any class
whose prediction looks unreliable -- based on how few points got predicted
for it and how confident the model actually was in those predictions.

This doesn't make a weak class (e.g. Camera) more accurate. It tells you
when to trust the output and when to check it by eye instead, based on
what we learned from the held-out evaluation: Battery and Screw have been
consistently reliable, Camera has not.

Usage:
    python predict_with_confidence.py --model pointnet2_holdout_test_v3.pt --mesh new_phone.obj

Typical Battery/Screw point counts and confidence, from the 5-phone holdout
set your thresholds default to, but adjust --min_points_* / --min_conf_*
if your own held-out numbers looked different.
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


def predict_with_confidence(model_path, mesh_path, num_points, device,
                             min_points, min_conf):
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

    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"Target mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    sampled_points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    sampled_normals = mesh.face_normals[face_idx]
    features = normalize_features_6d(sampled_points, sampled_normals)

    with torch.no_grad():
        input_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        preds = probs.argmax(axis=-1)

    print("\n" + "=" * 65)
    print(f" PREDICTION CONFIDENCE REPORT: {os.path.basename(mesh_path)}")
    print("=" * 65)

    results = {}
    for i, c_name in enumerate(class_names):
        mask = preds == i
        count = int(mask.sum())
        avg_conf = float(probs[mask, i].mean()) if count > 0 else 0.0

        p_min = min_points.get(c_name, 0)
        c_min = min_conf.get(c_name, 0.0)

        flags = []
        if count < p_min:
            flags.append(f"LOW POINT COUNT ({count} < expected min {p_min})")
        if count > 0 and avg_conf < c_min:
            flags.append(f"LOW CONFIDENCE ({avg_conf:.2f} < expected min {c_min:.2f})")
        if count == 0 and p_min > 0:
            flags.append("NOT DETECTED AT ALL")

        status = "OK" if not flags else "VERIFY MANUALLY -- " + "; ".join(flags)
        print(f"  {c_name:<10}: {count:>5} points, avg confidence {avg_conf:.3f}  -->  {status}")
        results[c_name] = {"count": count, "avg_conf": avg_conf, "flags": flags}

    print("=" * 65)
    any_flagged = any(r["flags"] for r in results.values())
    if any_flagged:
        print("\n>> One or more components need manual verification before trusting this result.")
    else:
        print("\n>> All components within expected confidence/count ranges.")

    return preds, probs, sampled_points, class_names, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--num_points", type=int, default=8192)
    # Defaults set from this project's own 5-phone holdout results: Battery/Screw
    # were reliably >100 points with reasonable confidence; Camera was the class
    # that kept failing, so it gets a stricter/more skeptical default.
    parser.add_argument("--min_points_battery", type=int, default=400)
    parser.add_argument("--min_points_camera", type=int, default=60)
    parser.add_argument("--min_points_screw", type=int, default=60)
    parser.add_argument("--min_conf_battery", type=float, default=0.6)
    parser.add_argument("--min_conf_camera", type=float, default=0.6)
    parser.add_argument("--min_conf_screw", type=float, default=0.6)
    args = parser.parse_args()

    min_points = {
        "Battery": args.min_points_battery,
        "Camera": args.min_points_camera,
        "Screw": args.min_points_screw,
    }
    min_conf = {
        "Battery": args.min_conf_battery,
        "Camera": args.min_conf_camera,
        "Screw": args.min_conf_screw,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_with_confidence(args.model, args.mesh, args.num_points, device, min_points, min_conf)

"""
Evaluate a trained checkpoint on the held-out test_split.npz produced
by make_split.py. Reports per-phone and aggregate per-point accuracy,
per-class accuracy, and a confusion matrix -- directly answers "does
it identify the held-out phones correctly", including whether the
Screw/Camera confusion persists on genuinely unseen data.

Usage:
    python evaluate_holdout.py --model pointnet2_holdout_test.pt --dataset test_split.npz

    # also dump colored point clouds (pred vs ground truth) per phone for visual sanity check
    python evaluate_holdout.py --model pointnet2_holdout_test.pt --dataset test_split.npz --export_ply --out_dir holdout_eval
"""

import argparse
import os
import numpy as np
import torch
import trimesh

from model import PointNetSegmentation
from model_pointnet2 import PointNet2Segmentation

CLASS_COLORS = {
    "Battery": [225, 30, 30, 255],
    "Camera": [30, 225, 30, 255],
    "Screw": [30, 90, 225, 255],
    "Other": [180, 180, 180, 255],
}


def load_model(model_path, device):
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
    return model, class_names, in_channels


def confusion_matrix(preds, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    header = "        " + "".join(f"{c[:8]:>10}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"{class_names[i][:8]:<8}{row_str}")


def export_colored_ply(points_xyz, labels, class_names, out_path):
    colors = np.array([CLASS_COLORS.get(class_names[c], [180, 180, 180, 255]) for c in labels], dtype=np.uint8)
    cloud = trimesh.points.PointCloud(vertices=points_xyz, colors=colors)
    cloud.export(out_path)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, in_channels = load_model(args.model, device)
    num_classes = len(class_names)
    print(f"Model: {args.model}  classes={class_names}  in_channels={in_channels}D  device={device}")

    data = np.load(args.dataset, allow_pickle=True)
    points_list = [np.asarray(p, dtype=np.float32) for p in data["points"]]
    labels_list = [np.asarray(l, dtype=np.int64) for l in data["labels"]]
    phone_names = list(data["phone_names"])

    if args.export_ply:
        os.makedirs(args.out_dir, exist_ok=True)

    all_preds_flat = []
    all_labels_flat = []
    per_phone_results = []

    for name, features, labels in zip(phone_names, points_list, labels_list):
        with torch.no_grad():
            input_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
            logits = model(input_tensor)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        acc = float((preds == labels).mean())
        cm = confusion_matrix(preds, labels, num_classes)

        per_class_acc = []
        for c in range(num_classes):
            mask = labels == c
            per_class_acc.append(float((preds[mask] == c).mean()) if mask.sum() > 0 else None)

        per_phone_results.append({
            "name": name, "acc": acc, "cm": cm, "per_class_acc": per_class_acc, "n_points": len(labels)
        })

        all_preds_flat.append(preds)
        all_labels_flat.append(labels)

        print(f"\n--- Phone {name} ({len(labels)} points) ---")
        print(f"Overall point accuracy: {acc:.4f}")
        for c, a in zip(class_names, per_class_acc):
            print(f"  {c:<8}: {'n/a (no gt points)' if a is None else f'{a:.4f}'}")
        print("Confusion matrix (rows=ground truth, cols=predicted):")
        print_confusion_matrix(cm, class_names)

        if args.export_ply:
            xyz = features[:, :3]
            export_colored_ply(xyz, labels, class_names, os.path.join(args.out_dir, f"{name}_groundtruth.ply"))
            export_colored_ply(xyz, preds, class_names, os.path.join(args.out_dir, f"{name}_predicted.ply"))

    all_preds_flat = np.concatenate(all_preds_flat)
    all_labels_flat = np.concatenate(all_labels_flat)
    overall_acc = float((all_preds_flat == all_labels_flat).mean())
    overall_cm = confusion_matrix(all_preds_flat, all_labels_flat, num_classes)

    print("\n" + "=" * 60)
    print(f"AGGREGATE OVER {len(phone_names)} HELD-OUT PHONES")
    print("=" * 60)
    print(f"Overall point accuracy: {overall_acc:.4f}")
    print("\nPer-phone accuracy:")
    for r in per_phone_results:
        print(f"  {r['name']:<10}: {r['acc']:.4f}")
    print("\nAggregate confusion matrix (rows=ground truth, cols=predicted):")
    print_confusion_matrix(overall_cm, class_names)

    print("\nAggregate per-class accuracy:")
    for c in range(num_classes):
        mask = all_labels_flat == c
        acc_c = float((all_preds_flat[mask] == c).mean()) if mask.sum() > 0 else None
        print(f"  {class_names[c]:<8}: {'n/a' if acc_c is None else f'{acc_c:.4f}'}  (n={int(mask.sum())})")

    if args.export_ply:
        print(f"\nColored point clouds saved to `{args.out_dir}/` "
              f"(*_groundtruth.ply vs *_predicted.ply per phone — open in MeshLab/CloudCompare to compare visually)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--dataset", required=True, help="Path to test_split.npz from make_split.py")
    parser.add_argument("--export_ply", action="store_true", help="Export colored ground-truth/predicted point clouds per phone")
    parser.add_argument("--out_dir", default="holdout_eval", help="Output folder for exported point clouds")
    args = parser.parse_args()

    evaluate(args)

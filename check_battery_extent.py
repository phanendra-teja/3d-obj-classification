"""
Checks Battery point-cloud extent (as a fraction of whole-phone extent)
across every phone in the dataset, to see if any phone has an unusually
large/small battery relative to the rest -- useful for diagnosing why
a specific phone's Battery class gets misclassified.

Usage:
    python check_battery_extent.py --dataset points_dataset_proportional.npz
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

data = np.load(args.dataset, allow_pickle=True)
points_list = [np.asarray(p, dtype=np.float32) for p in data["points"]]
labels_list = [np.asarray(l, dtype=np.int64) for l in data["labels"]]
phone_names = list(data["phone_names"])
class_names = list(data["class_names"])
battery_idx = class_names.index("Battery")

rows = []
for name, feats, labels in zip(phone_names, points_list, labels_list):
    xyz = feats[:, :3]
    phone_extent = xyz.max(axis=0) - xyz.min(axis=0)

    mask = labels == battery_idx
    if mask.sum() == 0:
        rows.append((name, None, None))
        continue

    battery_xyz = xyz[mask]
    battery_extent = battery_xyz.max(axis=0) - battery_xyz.min(axis=0)
    ratio = battery_extent / phone_extent
    avg_ratio = float(ratio.mean())
    rows.append((name, avg_ratio, mask.sum()))

print(f"{'Phone':<10}{'BatteryExtentRatio':<22}{'BatteryPoints':<15}")
ratios = [r[1] for r in rows if r[1] is not None]
mean_ratio = np.mean(ratios)
std_ratio = np.std(ratios)

for name, ratio, n in sorted(rows, key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0)):
    if ratio is None:
        print(f"{name:<10}{'no battery':<22}{'-':<15}")
        continue
    flag = "  <-- outlier" if abs(ratio - mean_ratio) > 2 * std_ratio else ""
    print(f"{name:<10}{ratio:<22.4f}{n:<15}{flag}")

print(f"\nMean ratio: {mean_ratio:.4f}, Std: {std_ratio:.4f}")

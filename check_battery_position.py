"""
Extent didn't explain phone 19's Battery failure (see check_battery_extent.py
output). This checks two other candidates:
  1. Where the battery centroid sits within the phone's bounding box
     (normalized 0-1 per axis) -- is 19 positioned atypically?
  2. The battery's dominant orientation (PCA principal axis, projected
     onto the phone's own principal axes) -- is 19 tilted/rotated
     atypically relative to the phone body?

Usage:
    python check_battery_position.py --dataset points_dataset_proportional.npz
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


def pca_axes(points):
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]  # columns = principal axes, descending eigenvalue


rows = []
for name, feats, labels in zip(phone_names, points_list, labels_list):
    xyz = feats[:, :3]
    mask = labels == battery_idx
    if mask.sum() == 0:
        rows.append((name, None, None, None))
        continue

    battery_xyz = xyz[mask]
    phone_min, phone_max = xyz.min(axis=0), xyz.max(axis=0)
    phone_range = phone_max - phone_min
    phone_range[phone_range == 0] = 1.0

    battery_centroid = battery_xyz.mean(axis=0)
    norm_pos = (battery_centroid - phone_min) / phone_range  # 0-1 per axis

    # Orientation: angle (deg) between phone's longest axis and battery's longest axis
    phone_axes = pca_axes(xyz)
    battery_axes = pca_axes(battery_xyz)
    cos_angle = np.abs(np.dot(phone_axes[:, 0], battery_axes[:, 0]))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    rows.append((name, norm_pos, angle_deg, mask.sum()))

print(f"{'Phone':<8}{'NormPos(x,y,z)':<28}{'AxisAngle(deg)':<18}{'BatteryPts':<12}")
positions = np.array([r[1] for r in rows if r[1] is not None])
angles = np.array([r[2] for r in rows if r[1] is not None])
mean_pos, std_pos = positions.mean(axis=0), positions.std(axis=0)
mean_angle, std_angle = angles.mean(), angles.std()

for name, pos, angle, n in rows:
    if pos is None:
        print(f"{name:<8}{'no battery':<28}")
        continue
    pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
    pos_outlier = np.any(np.abs(pos - mean_pos) > 2 * std_pos)
    angle_outlier = abs(angle - mean_angle) > 2 * std_angle
    flag = ""
    if pos_outlier:
        flag += "  <-- position outlier"
    if angle_outlier:
        flag += "  <-- orientation outlier"
    print(f"{name:<8}{pos_str:<28}{angle:<18.2f}{n:<12}{flag}")

print(f"\nMean normalized position: {mean_pos}, Std: {std_pos}")
print(f"Mean axis angle: {mean_angle:.2f} deg, Std: {std_angle:.2f} deg")

"""
Dataset Preparation for Whole-Mesh Component Classification
------------------------------------------------------------------
Converts your labeled component obj files (Battery/Camera/Screw per phone)
into point cloud + per-point label arrays suitable for training a
PointNet-style segmentation model.

EXPECTED FOLDER STRUCTURE:
    dataset/
        phone01/
            model_Battery.obj
            model_Camera.obj
            model_Screw.obj
        phone02/
            model_Battery.obj
            model_Camera.obj
            model_Screw.obj
        ...

One subfolder per phone. Component filenames must end in
"_<ComponentName>.obj" -- same convention as the Blender separation
script's output. Adjust CLASS_NAMES below if your taxonomy differs.

IMPORTANT LIMITATION: since your vertex groups only cover Battery/Camera/
Screw (not the whole phone), this trains a model that distinguishes
these 3 classes ONLY. It does not learn a general "background/other"
class for the rest of the phone. See the README for how to extend this
if you add a 4th "Other" vertex group later.

Usage:
    python prepare_dataset.py --dataset_dir dataset/ --output points_dataset.npz --points_per_component 512
"""

import argparse
import glob
import os
import re
import numpy as np
import trimesh

CLASS_NAMES = ["Battery", "Camera", "Screw"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
# Case-insensitive lookup: your exported filenames may not match the
# capitalization above exactly (e.g. "model_battery.obj" vs "Battery") --
# matching is done on lowercase so this doesn't silently drop data.
CLASS_TO_IDX_LOWER = {name.lower(): idx for name, idx in CLASS_TO_IDX.items()}


def component_name_from_filename(path):
    """'model_Battery.obj' -> 'Battery'"""
    base = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"^.*?_([A-Za-z0-9]+)$", base)
    return match.group(1) if match else base


def load_phone_point_cloud(phone_dir, points_per_component):
    """
    Loads all labeled component obj files in a phone's folder, samples a
    fixed number of points from each component's surface, and returns:
        points: (N, 3) float32
        labels: (N,) int64
    where N = points_per_component * number_of_components_found.
    """
    obj_paths = sorted(glob.glob(os.path.join(phone_dir, "*.obj")))
    all_points = []
    all_labels = []

    for path in obj_paths:
        component_name = component_name_from_filename(path)
        component_key = component_name.lower()
        if component_key not in CLASS_TO_IDX_LOWER:
            print(f"  WARNING: '{component_name}' (from {path}) not in CLASS_NAMES "
                  f"{CLASS_NAMES} -- skipping this file.")
            continue
        class_idx = CLASS_TO_IDX_LOWER[component_key]

        mesh = trimesh.load(path, force="mesh")
        if len(mesh.faces) == 0:
            print(f"  WARNING: {path} has no faces, skipping.")
            continue

        points, _ = trimesh.sample.sample_surface(mesh, points_per_component)
        labels = np.full(points.shape[0], class_idx, dtype=np.int64)

        all_points.append(points)
        all_labels.append(labels)

    if not all_points:
        return None, None

    return np.concatenate(all_points, axis=0).astype(np.float32), np.concatenate(all_labels, axis=0)


def normalize_point_cloud(points):
    """Centers on centroid and scales to unit sphere -- standard PointNet preprocessing."""
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    return points


def build_dataset(dataset_dir, points_per_component):
    phone_dirs = sorted(
        d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)
    )
    if not phone_dirs:
        raise FileNotFoundError(
            f"No phone subfolders found in {dataset_dir}. "
            f"Expected structure: {dataset_dir}/phone01/, {dataset_dir}/phone02/, ..."
        )

    print(f"Found {len(phone_dirs)} phone folder(s): {[os.path.basename(d) for d in phone_dirs]}\n")

    all_phone_points = []
    all_phone_labels = []
    phone_names = []
    incomplete_phones = []

    for phone_dir in phone_dirs:
        name = os.path.basename(phone_dir)
        print(f"Processing {name}...")
        points, labels = load_phone_point_cloud(phone_dir, points_per_component)
        if points is None:
            print(f"  WARNING: no valid components found for {name}, skipping this phone.")
            incomplete_phones.append((name, "no valid components"))
            continue

        points = normalize_point_cloud(points)
        class_counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))}
        print(f"  {points.shape[0]} points total, class counts: {class_counts}")

        missing = [c for c, count in class_counts.items() if count == 0]
        if missing:
            incomplete_phones.append((name, f"missing: {', '.join(missing)}"))

        all_phone_points.append(points)
        all_phone_labels.append(labels)
        phone_names.append(name)

    print("\n" + "=" * 60)
    if incomplete_phones:
        print(f"DATA QUALITY WARNING: {len(incomplete_phones)} phone(s) have missing classes:")
        for name, reason in incomplete_phones:
            print(f"  phone '{name}': {reason}")
        print("\nThese phones will still be included in the saved dataset, but training on")
        print("them will give the model zero supervision for whatever class is missing --")
        print("consider fixing the underlying Blender labeling, or re-run this script with")
        print("--exclude_incomplete to drop them from the dataset entirely.")
    else:
        print("All phones have all 3 classes present. No data quality issues detected.")
    print("=" * 60)

    return all_phone_points, all_phone_labels, phone_names, incomplete_phones


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Root folder containing one subfolder per phone")
    parser.add_argument("--output", default="points_dataset.npz")
    parser.add_argument("--points_per_component", type=int, default=512,
                         help="Number of points sampled from EACH component's surface")
    parser.add_argument("--exclude_incomplete", action="store_true",
                         help="Drop any phone missing one or more classes entirely from the saved dataset")
    args = parser.parse_args()

    all_points, all_labels, phone_names, incomplete_phones = build_dataset(args.dataset_dir, args.points_per_component)

    if args.exclude_incomplete and incomplete_phones:
        incomplete_names = {name for name, _ in incomplete_phones}
        keep_indices = [i for i, name in enumerate(phone_names) if name not in incomplete_names]
        dropped = len(phone_names) - len(keep_indices)
        all_points = [all_points[i] for i in keep_indices]
        all_labels = [all_labels[i] for i in keep_indices]
        phone_names = [phone_names[i] for i in keep_indices]
        print(f"\n--exclude_incomplete: dropped {dropped} phone(s), {len(phone_names)} remain.")

    # Saved as a ragged/object array since different phones may end up with
    # slightly different point counts if a component file is missing.
    np.savez(
        args.output,
        points=np.array(all_points, dtype=object),
        labels=np.array(all_labels, dtype=object),
        phone_names=np.array(phone_names),
        class_names=np.array(CLASS_NAMES),
    )
    print(f"\nSaved dataset for {len(phone_names)} phone(s) to {args.output}")
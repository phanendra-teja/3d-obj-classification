"""
Dataset Preparation with 6D Features (XYZ + Surface Normals)
------------------------------------------------------------
Converts labeled component OBJ files (Battery/Camera/Screw) into 6D point cloud
arrays (3D spatial coordinates + 3D surface normals) per phone.
"""

import argparse
import glob
import os
import re
import numpy as np
import trimesh

CLASS_NAMES = ["Battery", "Camera", "Screw"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX_LOWER = {name.lower(): idx for name, idx in CLASS_TO_IDX.items()}


def component_name_from_filename(path):
    """Extracts component name from filename like 'model_Battery.obj' -> 'Battery'."""
    base = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"^.*?_([A-Za-z0-9]+)$", base)
    return match.group(1) if match else base


def load_phone_point_cloud_with_normals(phone_dir, points_per_component):
    """Loads OBJ files and extracts surface points + surface normals."""
    obj_paths = sorted(glob.glob(os.path.join(phone_dir, "*.obj")))
    all_points = []
    all_normals = []
    all_labels = []

    for path in obj_paths:
        component_name = component_name_from_filename(path)
        component_key = component_name.lower()
        if component_key not in CLASS_TO_IDX_LOWER:
            continue
        class_idx = CLASS_TO_IDX_LOWER[component_key]

        mesh = trimesh.load(path, force="mesh")
        if len(mesh.faces) == 0:
            print(f"  WARNING: {path} has no faces, skipping.")
            continue

        # Sample points and get face indices to extract face normals
        points, face_indices = trimesh.sample.sample_surface(mesh, points_per_component)
        normals = mesh.face_normals[face_indices]
        labels = np.full(points.shape[0], class_idx, dtype=np.int64)

        all_points.append(points)
        all_normals.append(normals)
        all_labels.append(labels)

    if not all_points:
        return None, None

    points_arr = np.concatenate(all_points, axis=0).astype(np.float32)
    normals_arr = np.concatenate(all_normals, axis=0).astype(np.float32)
    labels_arr = np.concatenate(all_labels, axis=0)

    # Combine into 6D feature matrix (N, 6)
    features_arr = np.hstack([points_arr, normals_arr])
    return features_arr, labels_arr


def normalize_point_cloud_6d(features):
    """Centers coordinates on centroid, scales to unit sphere, and normalizes normals."""
    points = features[:, :3]
    normals = features[:, 3:]

    # Center and scale spatial coordinates
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist

    # Ensure surface normals are unit vectors
    norm_mags = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_mags[norm_mags == 0] = 1.0
    normals = normals / norm_mags

    return np.hstack([points, normals]).astype(np.float32)


def build_dataset(dataset_dir, points_per_component):
    phone_dirs = sorted(
        d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)
    )
    if not phone_dirs:
        raise FileNotFoundError(f"No phone subfolders found in {dataset_dir}.")

    print(f"Found {len(phone_dirs)} phone folder(s): {[os.path.basename(d) for d in phone_dirs]}\n")

    all_phone_features = []
    all_phone_labels = []
    phone_names = []
    incomplete_phones = []

    for phone_dir in phone_dirs:
        name = os.path.basename(phone_dir)
        print(f"Processing {name}...")
        features, labels = load_phone_point_cloud_with_normals(phone_dir, points_per_component)
        if features is None:
            print(f"  WARNING: no valid components found for {name}, skipping.")
            incomplete_phones.append((name, "no valid components"))
            continue

        features = normalize_point_cloud_6d(features)
        class_counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))}
        print(f"  {features.shape[0]} points total (6D features), class counts: {class_counts}")

        missing = [c for c, count in class_counts.items() if count == 0]
        if missing:
            incomplete_phones.append((name, f"missing: {', '.join(missing)}"))

        all_phone_features.append(features)
        all_phone_labels.append(labels)
        phone_names.append(name)

    print("\n" + "=" * 60)
    if incomplete_phones:
        print(f"DATA QUALITY WARNING: {len(incomplete_phones)} phone(s) have missing classes:")
        for name, reason in incomplete_phones:
            print(f"  phone '{name}': {reason}")
    else:
        print("All phones have all 3 classes present. No missing components detected.")
    print("=" * 60)

    return all_phone_features, all_phone_labels, phone_names, incomplete_phones


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Root folder containing one subfolder per phone")
    parser.add_argument("--output", default="points_dataset_normals.npz")
    parser.add_argument("--points_per_component", type=int, default=1024,
                        help="Number of points sampled from EACH component's surface")
    parser.add_argument("--exclude_incomplete", action="store_true",
                        help="Drop any phone missing one or more classes entirely from saved dataset")
    args = parser.parse_args()

    all_features, all_labels, phone_names, incomplete_phones = build_dataset(args.dataset_dir, args.points_per_component)

    if args.exclude_incomplete and incomplete_phones:
        incomplete_names = {name for name, _ in incomplete_phones}
        keep_indices = [i for i, name in enumerate(phone_names) if name not in incomplete_names]
        dropped = len(phone_names) - len(keep_indices)
        all_features = [all_features[i] for i in keep_indices]
        all_labels = [all_labels[i] for i in keep_indices]
        phone_names = [phone_names[i] for i in keep_indices]
        print(f"\n--exclude_incomplete: dropped {dropped} incomplete phone(s), {len(phone_names)} remain.")

    np.savez(
        args.output,
        points=np.array(all_features, dtype=object),
        labels=np.array(all_labels, dtype=object),
        phone_names=np.array(phone_names),
        class_names=np.array(CLASS_NAMES),
    )
    print(f"\nSaved 6D dataset for {len(phone_names)} phone(s) to {args.output}")
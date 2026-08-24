"""
Dataset Preparation for 4-Class Component & Chassis Segmentation
----------------------------------------------------------------
Ignores whole-mesh files (model_whole.obj) and extracts features
from individual component OBJ files.
"""

import argparse
import glob
import os
import re
import numpy as np
import trimesh

CLASS_NAMES = ["Battery", "Camera", "Screw", "Other"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX_LOWER = {name.lower(): idx for name, idx in CLASS_TO_IDX.items()}

OTHER_ALIASES = {"other", "chassis", "frame", "body", "whole_chassis"}


def component_name_from_filename(path):
    base = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"^.*?_([A-Za-z0-9]+)$", base)
    comp = match.group(1) if match else base
    
    if comp.lower() in OTHER_ALIASES:
        return "Other"
    return comp


def load_phone_point_cloud_with_normals(phone_dir, points_per_component):
    obj_paths = sorted(glob.glob(os.path.join(phone_dir, "*.obj")))
    all_points, all_normals, all_labels = [], [], []

    for path in obj_paths:
        filename = os.path.basename(path).lower()
        
        # Skip whole mesh file so it doesn't pollute class labels
        if "whole" in filename:
            continue

        comp_name = component_name_from_filename(path)
        comp_key = comp_name.lower()
        if comp_key not in CLASS_TO_IDX_LOWER:
            continue
        class_idx = CLASS_TO_IDX_LOWER[comp_key]

        mesh = trimesh.load(path, force="mesh")
        if len(mesh.faces) == 0:
            continue

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

    features_arr = np.hstack([points_arr, normals_arr])
    return features_arr, labels_arr


def normalize_point_cloud_6d(features):
    points = features[:, :3]
    normals = features[:, 3:]

    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist

    norm_mags = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_mags[norm_mags == 0] = 1.0
    normals = normals / norm_mags

    return np.hstack([points, normals]).astype(np.float32)


def build_dataset(dataset_dir, points_per_component):
    phone_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)])
    if not phone_dirs:
        raise FileNotFoundError(f"No phone subfolders found in {dataset_dir}.")

    all_phone_features, all_phone_labels, phone_names, incomplete_phones = [], [], [], []

    for phone_dir in phone_dirs:
        name = os.path.basename(phone_dir)
        features, labels = load_phone_point_cloud_with_normals(phone_dir, points_per_component)
        if features is None:
            incomplete_phones.append((name, "no valid OBJ components found"))
            continue

        features = normalize_point_cloud_6d(features)
        class_counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))}

        missing = [c for c, count in class_counts.items() if count == 0]
        print(f"Phone '{name}': counts = {class_counts}")
        if missing:
            incomplete_phones.append((name, f"missing: {', '.join(missing)}"))

        all_phone_features.append(features)
        all_phone_labels.append(labels)
        phone_names.append(name)

    return all_phone_features, all_phone_labels, phone_names, incomplete_phones


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", default="points_dataset_4class.npz")
    parser.add_argument("--points_per_component", type=int, default=1024)
    parser.add_argument("--exclude_incomplete", action="store_true")
    args = parser.parse_args()

    all_features, all_labels, phone_names, incomplete_phones = build_dataset(args.dataset_dir, args.points_per_component)

    if incomplete_phones:
        print("\n" + "=" * 60)
        print(f"Incomplete phones ({len(incomplete_phones)} total):")
        for name, reason in incomplete_phones:
            print(f"  Phone {name}: {reason}")
        print("=" * 60)

    if args.exclude_incomplete and incomplete_phones:
        incomplete_names = {name for name, _ in incomplete_phones}
        keep_indices = [i for i, name in enumerate(phone_names) if name not in incomplete_names]
        dropped = len(phone_names) - len(keep_indices)
        all_features = [all_features[i] for i in keep_indices]
        all_labels = [all_labels[i] for i in keep_indices]
        phone_names = [phone_names[i] for i in keep_indices]
        print(f"\n--exclude_incomplete: dropped {dropped} phone(s), {len(phone_names)} valid phone(s) remain.")

    np.savez(
        args.output,
        points=np.array(all_features, dtype=object),
        labels=np.array(all_labels, dtype=object),
        phone_names=np.array(phone_names),
        class_names=np.array(CLASS_NAMES),
    )
    print(f"\nSaved 4-class dataset ({len(phone_names)} phones) to {args.output}")
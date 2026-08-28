"""
Dataset Preparation: Proportional Scene Sampling for 4-Class Phone Segmentation
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


def load_whole_phone_scene(phone_dir, total_points=8192):
    obj_paths = sorted(glob.glob(os.path.join(phone_dir, "*.obj")))
    valid_paths = [p for p in obj_paths if "whole" not in os.path.basename(p).lower()]

    component_meshes = []
    component_labels = []

    for path in valid_paths:
        comp_name = component_name_from_filename(path)
        comp_key = comp_name.lower()
        if comp_key not in CLASS_TO_IDX_LOWER:
            continue

        c_idx = CLASS_TO_IDX_LOWER[comp_key]
        mesh = trimesh.load(path, force="mesh")
        if len(mesh.faces) == 0:
            continue

        component_meshes.append(mesh)
        component_labels.append(c_idx)

    if not component_meshes:
        return None, None

    # Calculate area proportional sampling rates
    areas = np.array([m.area for m in component_meshes])
    total_area = areas.sum()
    if total_area == 0:
        return None, None

    sampled_pts_list, sampled_norms_list, sampled_lbls_list = [], [], []

    for mesh, c_idx, area in zip(component_meshes, component_labels, areas):
        # Guarantee minimum samples for tiny components like screws/cameras
        n_pts = max(128, int(round((area / total_area) * total_points)))
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_pts)
        norms = mesh.face_normals[face_idx]
        lbls = np.full(n_pts, c_idx, dtype=np.int64)

        sampled_pts_list.append(pts)
        sampled_norms_list.append(norms)
        sampled_lbls_list.append(lbls)

    pts_arr = np.concatenate(sampled_pts_list, axis=0)
    norms_arr = np.concatenate(sampled_norms_list, axis=0)
    lbls_arr = np.concatenate(sampled_lbls_list, axis=0)

    # Resample or truncate to exact total_points
    if len(pts_arr) > total_points:
        idx = np.random.choice(len(pts_arr), total_points, replace=False)
    else:
        idx = np.random.choice(len(pts_arr), total_points, replace=True)

    pts_arr = pts_arr[idx]
    norms_arr = norms_arr[idx]
    lbls_arr = lbls_arr[idx]

    # Normalize 6D spatial coordinates
    centroid = pts_arr.mean(axis=0)
    pts_norm = pts_arr - centroid
    max_d = np.max(np.linalg.norm(pts_norm, axis=1))
    if max_d > 0:
        pts_norm = pts_norm / max_d

    norm_mags = np.linalg.norm(norms_arr, axis=1, keepdims=True)
    norm_mags[norm_mags == 0] = 1.0
    norms_norm = norms_arr / norm_mags

    features = np.hstack([pts_norm, norms_norm]).astype(np.float32)
    return features, lbls_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--output", default="points_dataset_proportional.npz")
    parser.add_argument("--total_points", type=int, default=8192)
    args = parser.parse_args()

    phone_dirs = sorted([d for d in glob.glob(os.path.join(args.dataset_dir, "*")) if os.path.isdir(d)])
    all_feats, all_lbls, names = [], [], []

    for pdir in phone_dirs:
        pname = os.path.basename(pdir)
        feats, lbls = load_whole_phone_scene(pdir, total_points=args.total_points)
        if feats is None:
            continue

        counts = {CLASS_NAMES[i]: int((lbls == i).sum()) for i in range(len(CLASS_NAMES))}
        print(f"Phone '{pname}': {counts}")

        all_feats.append(feats)
        all_lbls.append(lbls)
        names.append(pname)

    np.savez(
        args.output,
        points=np.array(all_feats, dtype=object),
        labels=np.array(all_lbls, dtype=object),
        phone_names=np.array(names),
        class_names=np.array(CLASS_NAMES),
    )
    print(f"\nSaved proportional scene dataset ({len(names)} phones) to `{args.output}`")
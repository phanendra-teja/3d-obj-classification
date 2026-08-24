"""
Generate clean model_Other.obj via memory-efficient k-d tree face filtering.
"""

import os
import glob
import numpy as np
import trimesh
from scipy.spatial import cKDTree

DATASET_DIR = "dataset"
phone_dirs = sorted([d for d in glob.glob(os.path.join(DATASET_DIR, "*")) if os.path.isdir(d)])

print(f"Processing {len(phone_dirs)} phone directories...")

for pdir in phone_dirs:
    phone_name = os.path.basename(pdir)
    obj_files = glob.glob(os.path.join(pdir, "*.obj"))

    whole_files = [f for f in obj_files if "whole" in os.path.basename(f).lower()]
    comp_files = [f for f in obj_files if "whole" not in os.path.basename(f).lower() and "other" not in os.path.basename(f).lower()]

    if not whole_files or not comp_files:
        continue

    whole_mesh = trimesh.load(whole_files[0], force="mesh")
    comp_meshes = [trimesh.load(f, force="mesh") for f in comp_files if len(trimesh.load(f, force="mesh").faces) > 0]

    if not comp_meshes:
        continue

    combined_comps = trimesh.util.concatenate(comp_meshes)

    # Sample component surface points to build spatial k-d tree
    comp_pts, _ = trimesh.sample.sample_surface(combined_comps, 20000)
    tree = cKDTree(comp_pts)

    # Find whole_mesh vertices further than 1.5mm from any labeled component
    distances, _ = tree.query(whole_mesh.vertices)
    vertex_mask = distances > 0.0015

    # Map vertex mask to face mask (keep face if all 3 vertices belong to chassis)
    face_mask = vertex_mask[whole_mesh.faces].all(axis=1)

    # Extract clean chassis submesh
    chassis_mesh = whole_mesh.submesh([face_mask], append=True)
    out_path = os.path.join(pdir, "model_Other.obj")
    chassis_mesh.export(out_path)
    print(f"  Phone {phone_name}: Clean model_Other.obj exported successfully.")

print("\nChassis generation complete across all folders!")
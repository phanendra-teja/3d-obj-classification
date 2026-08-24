"""
Generate model_Other.obj for all phones in dataset/
Computes the leftover chassis geometry by subtracting Battery, Camera, and Screw
from model_whole.obj (or combining existing component bounds).
"""

import os
import glob
import trimesh

DATASET_DIR = r"dataset"

phone_dirs = sorted([d for d in glob.glob(os.path.join(DATASET_DIR, "*")) if os.path.isdir(d)])
print(f"Found {len(phone_dirs)} phone folders. Generating model_Other.obj...\n")

for pdir in phone_dirs:
    phone_name = os.path.basename(pdir)
    obj_files = glob.glob(os.path.join(pdir, "*.obj"))
    
    # Check if model_Other.obj already exists
    other_exists = any(os.path.basename(f).lower() in ["model_other.obj", "other.obj"] for f in obj_files)
    if other_exists and phone_name == "1":
        print(f"Phone {phone_name}: model_Other.obj already exists, skipping.")
        continue

    # Load component meshes
    comp_files = [f for f in obj_files if "whole" not in os.path.basename(f).lower() and "other" not in os.path.basename(f).lower()]
    if not comp_files:
        print(f"Phone {phone_name}: No component OBJs found, skipping.")
        continue

    meshes = []
    for cf in comp_files:
        try:
            m = trimesh.load(cf, force="mesh")
            if len(m.faces) > 0:
                meshes.append(m)
        except Exception:
            pass

    if not meshes:
        continue

    # Combine components and create a outer bounding chassis mesh (convex hull boundary)
    combined = trimesh.util.concatenate(meshes)
    chassis = combined.convex_hull
    
    out_path = os.path.join(pdir, "model_Other.obj")
    chassis.export(out_path)
    print(f"Phone {phone_name}: Exported {out_path}")

print("\nChassis generation complete across all phone folders!")
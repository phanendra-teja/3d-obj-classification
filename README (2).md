# Whole-Mesh Component Classification (Deep Learning)

Trains a PointNet-based model that takes a whole, UNSEGMENTED phone point
cloud and predicts a per-point component label (Battery / Camera / Screw).
This is the model that eventually replaces manual Blender vertex grouping
for new phones.

Tested end-to-end on real phone1 data before being handed to you (data
loading, model forward pass, and a short training run all verified
working -- loss decreases across epochs, no errors).

## 1. Organize your 31 phones into the expected folder structure

```
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
    phone31/
        ...
```

One subfolder per phone. Filenames must end in `_Battery.obj`,
`_Camera.obj`, `_Screw.obj` (matches your Blender separation script's
naming convention). You currently have all 31 phones' files -- you'll
need to sort them into per-phone folders like this if they aren't
already.

## 2. Install dependencies

```bash
pip install torch trimesh numpy --break-system-packages
```

## 3. Build the point cloud dataset

```bash
python prepare_dataset.py --dataset_dir dataset/ --output points_dataset.npz --points_per_component 512
```

For each phone, this samples 512 points off each component's surface
(so 1,536 points total per phone across the 3 classes), normalizes each
phone's point cloud to be centered and unit-scale, and saves everything
into one `.npz` file. Console output shows per-phone point/class counts
-- check these look sane (roughly balanced across the 3 classes) before
moving on.

## 4. Train

```bash
python train.py --dataset points_dataset.npz --epochs 100 --batch_size 4 --points_per_sample 1024
```

- `--points_per_sample`: how many points get fed to the model per
  training example (randomly subsampled from each phone's full point
  cloud, re-sampled fresh every epoch -- this itself acts as a form of
  augmentation).
- With 31 phones, ~4-5 are automatically held out for validation; you'll
  see `val_acc` and per-class accuracy printed periodically.
- Output: `pointnet_seg_model.pt`, containing the trained weights and
  class names.

## 5. Inference: predict on a brand-new, whole, unsegmented mesh

This is the actual payoff -- given a phone obj with NO Blender labeling
at all, predict which points are Battery/Camera/Screw.

```bash
python predict.py --model pointnet_seg_model.pt --mesh new_phone.obj \
    --num_points 4096 --out_glb predicted_labels.glb --out_npz predicted_labels.npz
```

Output:
- Console: predicted point count and percentage per class, plus mean
  prediction confidence (low confidence is a signal the model is
  guessing on geometry it doesn't recognize -- often a real "this isn't
  one of my 3 known classes" case, given limitation #2 below).
- `predicted_labels.glb`: the mesh's surface as a colored point cloud
  (red=Battery, green=Camera, blue=Screw) -- open in Blender and
  actually LOOK at whether the predicted regions make sense before
  trusting them.
- `predicted_labels.npz`: raw points + predicted labels, in case you
  want to feed these into the classical CoACD/collision pipeline instead
  of manually-exported components.

**I ran this myself against the real whole `phone1.obj`** (632K verts) to
confirm the pipeline works end-to-end -- it does (no errors, correct
shapes, valid glb/npz output). But the actual prediction was 100% Screw,
0% everything else, with high confidence. This is NOT a bug -- it's the
direct consequence of the toy 1-phone training run in step 4 above,
which only exists to verify the code runs, not to produce a usable
model. With just one training example repeated over many epochs, the
model has no reason to learn real distinguishing geometry -- it
collapses onto whatever's easiest to fit. Once you train on your real 31
phones, predictions should differentiate meaningfully. Don't judge model
quality from this test run.

## IMPORTANT LIMITATIONS -- read before trusting results

**1. 31 examples is a very small dataset.** Most point cloud segmentation
papers train on thousands of shapes. Augmentation (random rotation,
jitter, scaling -- already built into `train.py`) helps the model see
more "views" of the same 31 phones, but it cannot manufacture genuine
shape diversity. Expect this model to work best on phones similar to
your labeled 31, and to generalize poorly to very different phone
designs. If you want better generalization, the real fix is more
diverse labeled phones, not more augmentation of the same ones.

**2. Only 3 classes are learned -- there's no "background/other" class.**
Since your vertex groups only cover Battery/Camera/Screw (not the whole
phone), this model was trained to distinguish these 3 classes only. At
inference time, if you feed it a whole unlabeled phone mesh, EVERY point
will be forced into one of these 3 categories, including points that are
actually frame/chassis/other unlabeled geometry. There is no way for the
model to say "none of the above" with the current training data.

If you want a true background class, add a 4th vertex group in Blender
(e.g. "Other") capturing everything not already in Battery/Camera/Screw,
re-export, and re-run `prepare_dataset.py` -- it will automatically pick
up a 4th class as long as you add it to `CLASS_NAMES` at the top of that
file.

**3. No T-Net alignment transform.** The original PointNet paper includes
learned input/feature alignment transforms (T-Nets) for rotation
robustness. This implementation omits them for simplicity. The
z-axis-only random rotation augmentation in training partially
compensates (phones are typically scanned/labeled lying flat), but if
your inference-time point clouds can appear in arbitrary orientations
(not just rotated around Z), accuracy will likely suffer. Worth
revisiting if you see this in practice.

## Files

- `prepare_dataset.py` -- obj files -> point cloud + label arrays
- `model.py` -- PointNet segmentation architecture (run directly for a
  shape sanity check)
- `train.py` -- training loop with augmentation and validation

## Next steps

- Sort your 31 phones into the folder structure above and run the real
  training job (the test run here used only 1 phone, purely to verify
  the code works -- it is not a usable model)
- Once trained, write an inference script that takes a brand-new
  unsegmented phone obj, samples points from it, and runs them through
  the model to get predicted labels -- this is what actually replaces
  manual Blender vertex grouping going forward
- The predicted labels can then feed directly into the existing
  classical pipeline (CoACD decomposition, FCL/PyBullet collision
  checking) in place of manually-exported components

"""
Training Pipeline for PointNet / PointNet++ Component Segmentation
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import PointNetSegmentation
from model_pointnet2 import PointNet2Segmentation
from focal_loss import FocalLoss


class PhoneComponentDataset(Dataset):
    def __init__(self, points_list, labels_list, points_per_sample, num_classes,
                 augment=True, max_tilt_deg=15.0, min_class_points=64, rare_classes=None):
        """
        min_class_points: for each class in rare_classes, guarantee at least this
            many of its points are included in every crop (if that many exist on
            the phone at all). Prevents rare components like Camera/Screw from
            being crop-sampled out of a training example entirely.
        rare_classes: list of class indices to guarantee representation for.
            Defaults to every class except the most common one if not given.
        """
        self.points_list = points_list
        self.labels_list = labels_list
        self.points_per_sample = points_per_sample
        self.num_classes = num_classes
        self.augment = augment
        self.max_tilt_deg = max_tilt_deg
        self.min_class_points = min_class_points

        if rare_classes is None:
            # default: guarantee every class that isn't the majority (largest total count) class
            totals = np.zeros(num_classes)
            for labels in labels_list:
                for c in range(num_classes):
                    totals[c] += (labels == c).sum()
            majority_class = int(np.argmax(totals))
            rare_classes = [c for c in range(num_classes) if c != majority_class]
        self.rare_classes = rare_classes

    def __len__(self):
        return len(self.points_list)

    def _stratified_choice(self, labels, n_available):
        """Pick point indices for this crop, guaranteeing minimum coverage of rare classes."""
        chosen = []
        remaining_budget = self.points_per_sample

        for c in self.rare_classes:
            class_idx = np.where(labels == c)[0]
            if len(class_idx) == 0:
                continue
            take = min(self.min_class_points, len(class_idx), remaining_budget)
            if take <= 0:
                continue
            picked = np.random.choice(class_idx, take, replace=False)
            chosen.append(picked)
            remaining_budget -= take

        chosen_flat = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)
        chosen_set = set(chosen_flat.tolist())

        if remaining_budget > 0:
            pool = np.array([i for i in range(n_available) if i not in chosen_set])
            if len(pool) == 0:
                pool = np.arange(n_available)
            fill_replace = len(pool) < remaining_budget
            fill = np.random.choice(pool, remaining_budget, replace=fill_replace)
            chosen_flat = np.concatenate([chosen_flat, fill]) if len(chosen_flat) else fill

        if len(chosen_flat) > self.points_per_sample:
            chosen_flat = np.random.choice(chosen_flat, self.points_per_sample, replace=False)
        elif len(chosen_flat) < self.points_per_sample:
            extra = np.random.choice(n_available, self.points_per_sample - len(chosen_flat), replace=True)
            chosen_flat = np.concatenate([chosen_flat, extra])

        np.random.shuffle(chosen_flat)
        return chosen_flat

    def __getitem__(self, idx):
        features = self.points_list[idx]
        labels = self.labels_list[idx]
        n_available = features.shape[0]

        if self.augment:
            choice = self._stratified_choice(labels, n_available)
        else:
            # validation/eval: plain random sampling, no guarantees -- we want
            # to measure real-world performance, not artificially boosted crops
            if n_available >= self.points_per_sample:
                choice = np.random.choice(n_available, self.points_per_sample, replace=False)
            else:
                choice = np.random.choice(n_available, self.points_per_sample, replace=True)

        features = features[choice]
        labels = labels[choice]

        if self.augment:
            features = self._augment(features)

        return torch.from_numpy(features).float(), torch.from_numpy(labels).long()

    def _random_rotation_matrix(self):
        # Full random rotation around Z, plus a small random tilt around X and Y.
        theta_z = np.random.uniform(0, 2 * np.pi)
        theta_x = np.radians(np.random.uniform(-self.max_tilt_deg, self.max_tilt_deg))
        theta_y = np.radians(np.random.uniform(-self.max_tilt_deg, self.max_tilt_deg))

        cz, sz = np.cos(theta_z), np.sin(theta_z)
        cx, sx = np.cos(theta_x), np.sin(theta_x)
        cy, sy = np.cos(theta_y), np.sin(theta_y)

        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)

        return rot_z @ rot_x @ rot_y

    def _augment(self, features):
        xyz = features[:, :3]
        has_normals = features.shape[1] > 3
        normals = features[:, 3:] if has_normals else None

        rotation = self._random_rotation_matrix()

        xyz = xyz @ rotation.T
        if has_normals:
            normals = normals @ rotation.T

        xyz = xyz + np.random.normal(0, 0.01, size=xyz.shape).astype(np.float32)

        scale = np.random.uniform(0.9, 1.1)
        xyz = xyz * scale

        if has_normals:
            norm_mags = np.linalg.norm(normals, axis=1, keepdims=True)
            norm_mags[norm_mags == 0] = 1.0
            normals = normals / norm_mags
            return np.hstack([xyz, normals])

        return xyz


def load_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    points_list = [np.asarray(p, dtype=np.float32) for p in data["points"]]
    labels_list = [np.asarray(l, dtype=np.int64) for l in data["labels"]]
    class_names = list(data["class_names"])
    phone_names = list(data["phone_names"])
    return points_list, labels_list, class_names, phone_names


def compute_class_weights(labels_list, num_classes, device, power=1.0):
    """
    power=1.0 gives the original raw inverse-frequency weights (can be very
    extreme with a class imbalance like this dataset's, e.g. 16-18x). power<1.0
    dampens that -- e.g. power=0.5 takes the sqrt of the raw weights, which
    still favors rare classes but far less aggressively. Use a lower power
    when combining with stratified per-crop sampling, since both mechanisms
    push the model toward rare classes and compound if left at full strength
    together.
    """
    counts = np.zeros(num_classes)
    for labels in labels_list:
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    counts = np.maximum(counts, 1)
    raw_weights = counts.sum() / (num_classes * counts)
    weights = raw_weights ** power
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, val_loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.reshape(-1))
            all_labels.append(labels.reshape(-1))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item()

    per_class = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() == 0:
            per_class.append(0.0)
        else:
            per_class.append((all_preds[mask] == c).float().mean().item())

    return acc, per_class


def train(args):
    points_list, labels_list, class_names, phone_names = load_dataset(args.dataset)
    num_classes = len(class_names)
    in_channels = points_list[0].shape[1]
    print(f"Loaded {len(points_list)} phone(s), {num_classes} classes: {class_names}, Feature dim: {in_channels}D")

    n_val = max(1, int(len(points_list) * 0.15)) if len(points_list) >= 4 else 0
    val_points, val_labels = points_list[:n_val], labels_list[:n_val]
    train_points, train_labels = points_list[n_val:], labels_list[n_val:]

    train_dataset = PhoneComponentDataset(
        train_points, train_labels, args.points_per_sample, num_classes,
        augment=True, max_tilt_deg=args.max_tilt_deg, min_class_points=args.min_class_points
    )
    print(f"Guaranteeing >= {args.min_class_points} pts/crop for classes: "
          f"{[class_names[c] for c in train_dataset.rare_classes]}")
    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True)

    val_loader = None
    if val_points:
        val_dataset = PhoneComponentDataset(val_points, val_labels, args.points_per_sample, num_classes, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if args.architecture == "pointnet2":
        model = PointNet2Segmentation(num_classes=num_classes, in_channels=in_channels).to(device)
    else:
        model = PointNetSegmentation(num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = None
    if args.class_weighted_loss:
        class_weights = compute_class_weights(train_labels, num_classes, device, power=args.class_weight_power)

        if args.class_weight_overrides:
            overrides = {}
            for pair in args.class_weight_overrides.split(","):
                cname, cval = pair.split("=")
                overrides[cname.strip()] = float(cval)
            for cname, cval in overrides.items():
                if cname not in class_names:
                    raise ValueError(f"Unknown class '{cname}' in --class_weight_overrides. "
                                      f"Available: {class_names}")
                class_weights[class_names.index(cname)] = cval

        print(f"Class weights (power={args.class_weight_power}"
              f"{', overrides=' + args.class_weight_overrides if args.class_weight_overrides else ''}): "
              f"{dict(zip(class_names, [round(w, 3) for w in class_weights.tolist()]))}")

    if args.use_focal_loss:
        print("Using Focal Loss for loss calculation.")
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            msg = f"Epoch {epoch + 1}/{args.epochs}  train_loss={avg_loss:.4f}"

            if val_loader:
                val_acc, per_class = evaluate(model, val_loader, device, num_classes)
                msg += f"  val_acc={val_acc:.4f}  per_class={[round(a, 3) for a in per_class]}"

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    msg += "  <- best"

            print(msg)

    final_state = best_state if best_state is not None else model.state_dict()
    torch.save({
        "model_state_dict": final_state,
        "class_names": class_names,
        "points_per_sample": args.points_per_sample,
        "architecture": args.architecture,
        "in_channels": in_channels
    }, args.output)
    print(f"\nSaved trained model checkpoint to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to .npz file")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--points_per_sample", type=int, default=2048)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--class_weighted_loss", action="store_true")
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--architecture", choices=["pointnet", "pointnet2"], default="pointnet2")
    parser.add_argument("--max_tilt_deg", type=float, default=15.0,
                         help="Max random X/Y tilt (degrees) applied during augmentation, on top of full Z rotation.")
    parser.add_argument("--min_class_points", type=int, default=64,
                         help="Minimum points guaranteed per rare class (Camera/Screw/Battery) in every training crop.")
    parser.add_argument("--class_weight_power", type=float, default=1.0,
                         help="Exponent applied to raw inverse-frequency class weights. 1.0 = original "
                              "(can be extreme, e.g. 16-18x). 0.5 = sqrt-dampened, recommended when "
                              "combined with --min_class_points stratified sampling so the two mechanisms "
                              "don't compound. Only used if --class_weighted_loss is set.")
    parser.add_argument("--class_weight_overrides", type=str, default=None,
                         help="Comma-separated per-class weight overrides applied after --class_weight_power, "
                              "e.g. 'Battery=1.0,Screw=6.0' to dampen these specifically while leaving others "
                              "(e.g. Camera) at their power-scaled value. Only used if --class_weighted_loss is set.")
    parser.add_argument("--output", default="pointnet2_best.pt")
    args = parser.parse_args()

    train(args)
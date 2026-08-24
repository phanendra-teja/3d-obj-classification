"""
Training Script: Whole-Mesh Component Classification
------------------------------------------------------------
Trains the PointNet segmentation model on the dataset produced by
prepare_dataset.py.

IMPORTANT ABOUT DATASET SIZE: with 31 phones total, this is a very small
dataset by deep learning standards -- most point cloud segmentation
papers train on thousands of shapes. To get anything useful out of this
few examples, AUGMENTATION IS NOT OPTIONAL here -- every training sample
is randomly rotated, jittered, and re-sampled (a different random subset
of points) EVERY EPOCH, so the model sees many different "views" of the
same 31 underlying phones rather than memorizing 31 fixed point sets.

Even with heavy augmentation, expect this to generalize best to phones
similar to your labeled 31, not to wildly different phone designs -- that
would need more diverse training phones, not just more augmentation of
the same ones.

Usage:
    python train.py --dataset points_dataset.npz --epochs 100 --points_per_sample 1024
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import PointNetSegmentation
from model_pointnet2 import PointNet2Segmentation


class PhoneComponentDataset(Dataset):
    def __init__(self, points_list, labels_list, points_per_sample, augment=True):
        self.points_list = points_list
        self.labels_list = labels_list
        self.points_per_sample = points_per_sample
        self.augment = augment

    def __len__(self):
        return len(self.points_list)

    def __getitem__(self, idx):
        points = self.points_list[idx]
        labels = self.labels_list[idx]

        # Random subsample (or upsample with replacement) to a fixed size
        n_available = points.shape[0]
        if n_available >= self.points_per_sample:
            choice = np.random.choice(n_available, self.points_per_sample, replace=False)
        else:
            choice = np.random.choice(n_available, self.points_per_sample, replace=True)
        points = points[choice]
        labels = labels[choice]

        if self.augment:
            points = self._augment(points)

        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()

    def _augment(self, points):
        # Random rotation about the Z axis (phones are typically laid flat)
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        points = points @ rotation.T

        # Small Gaussian jitter per point
        points = points + np.random.normal(0, 0.01, size=points.shape).astype(np.float32)

        # Small random uniform scale
        scale = np.random.uniform(0.9, 1.1)
        points = points * scale

        # Random point dropout: zero out a random fraction of points (replacing
        # them with a duplicate of a kept point rather than removing them, so
        # the tensor shape stays fixed). Standard point-cloud augmentation --
        # forces the model to rely on more than any single distinctive point,
        # which helps generalization on a dataset this small.
        if np.random.random() < 0.5:  # only apply this augmentation ~half the time
            dropout_ratio = np.random.uniform(0.0, 0.2)
            n_points = points.shape[0]
            drop_idx = np.where(np.random.random(n_points) < dropout_ratio)[0]
            if len(drop_idx) > 0:
                points[drop_idx] = points[0]  # collapse dropped points onto the first point

        return points


def load_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    points_list = [np.asarray(p, dtype=np.float32) for p in data["points"]]
    labels_list = [np.asarray(l, dtype=np.int64) for l in data["labels"]]
    class_names = list(data["class_names"])
    phone_names = list(data["phone_names"])
    return points_list, labels_list, class_names, phone_names


def compute_per_class_accuracy(preds, labels, num_classes):
    accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == c).float().mean().item())
    return accs


def compute_confusion_matrix(preds, labels, num_classes):
    """Rows = true class, columns = predicted class."""
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        matrix[t, p] += 1
    return matrix


def print_confusion_matrix(matrix, class_names):
    print("\nConfusion matrix (rows=true class, columns=predicted class):")
    header = "".join(f"{name[:10]:>12}" for name in class_names)
    print(f"{'':>12}{header}")
    for i, name in enumerate(class_names):
        row = matrix[i]
        total = row.sum().item()
        row_str = "".join(f"{row[j].item():>12}" for j in range(len(class_names)))
        print(f"{name[:10]:>12}{row_str}   (n={total})")


def evaluate(model, val_loader, device, num_classes, num_passes=3, return_confusion=False):
    """
    Runs multiple forward passes over the validation set (each with a
    different random point subsample, since PhoneComponentDataset
    resamples every call) and averages the result. With a validation set
    this small (a handful of phones), a single pass is noisy enough to
    be misleading -- averaging several passes gives a more trustworthy
    number without needing more validation phones.
    """
    model.eval()
    accs = []
    per_class_accs = []
    last_preds, last_labels = None, None
    with torch.no_grad():
        for _ in range(num_passes):
            all_preds, all_labels = [], []
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                logits = model(points)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.reshape(-1))
                all_labels.append(labels.reshape(-1))
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            accs.append((all_preds == all_labels).float().mean().item())
            per_class_accs.append(compute_per_class_accuracy(all_preds, all_labels, num_classes))
            last_preds, last_labels = all_preds, all_labels

    avg_acc = sum(accs) / len(accs)
    max_len = max(len(pc) for pc in per_class_accs)
    avg_per_class = []
    for c in range(max_len):
        vals = [pc[c] for pc in per_class_accs if len(pc) > c]
        avg_per_class.append(sum(vals) / len(vals) if vals else float("nan"))

    if return_confusion:
        confusion = compute_confusion_matrix(last_preds.cpu(), last_labels.cpu(), num_classes)
        return avg_acc, avg_per_class, confusion
    return avg_acc, avg_per_class


def compute_class_weights(labels_list, num_classes, device):
    """
    Inverse-frequency class weights for the loss function. Screw is
    consistently the hardest class to learn (small, scattered geometry
    vs. Battery/Camera's large contiguous regions) -- upweighting its
    contribution to the loss pushes the model to prioritize getting it
    right instead of settling for high accuracy on the two easy classes.
    """
    counts = np.zeros(num_classes)
    for labels in labels_list:
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    counts = np.maximum(counts, 1)  # avoid divide-by-zero
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(args):
    points_list, labels_list, class_names, phone_names = load_dataset(args.dataset)
    num_classes = len(class_names)
    print(f"Loaded {len(points_list)} phone(s), {num_classes} classes: {class_names}")

    if len(points_list) < 3:
        print(f"\nWARNING: only {len(points_list)} phone(s) in the dataset. "
              f"Training/validation split will be minimal or skipped. "
              f"This run is only useful for verifying the pipeline runs correctly, "
              f"not for producing a usable model -- add more phones before real training.\n")

    # Simple split: hold out ~15% of phones for validation, if enough exist
    n_val = max(1, int(len(points_list) * 0.15)) if len(points_list) >= 4 else 0
    val_points, val_labels = points_list[:n_val], labels_list[:n_val]
    train_points, train_labels = points_list[n_val:], labels_list[n_val:]

    train_dataset = PhoneComponentDataset(train_points, train_labels, args.points_per_sample, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True)

    val_loader = None
    if val_points:
        val_dataset = PhoneComponentDataset(val_points, val_labels, args.points_per_sample, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = (PointNet2Segmentation(num_classes=num_classes) if args.architecture == "pointnet2"
              else PointNetSegmentation(num_classes=num_classes)).to(device)
    print(f"Architecture: {args.architecture}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.class_weighted_loss:
        class_weights = compute_class_weights(train_labels, num_classes, device)
        print(f"Using class-weighted loss. Weights: "
              f"{dict(zip(class_names, [round(w, 3) for w in class_weights.tolist()]))}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(points)  # (batch, N, num_classes)
            loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            msg = f"Epoch {epoch + 1}/{args.epochs}  train_loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"

            if val_loader:
                overall_acc, per_class = evaluate(model, val_loader, device, num_classes, num_passes=3)
                msg += f"  val_acc={overall_acc:.4f} (avg of 3 passes)  per_class={[round(a, 3) for a in per_class]}"

                if overall_acc > best_val_acc:
                    best_val_acc = overall_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    msg += "  <- new best"

            print(msg)

    final_state = best_state if best_state is not None else model.state_dict()
    if best_state is not None:
        print(f"\nUsing BEST checkpoint (val_acc={best_val_acc:.4f}), not necessarily the final epoch's "
              f"weights -- the final epoch is not guaranteed to be the best given how noisy validation "
              f"accuracy is with this few phones.")

        model.load_state_dict(best_state)
        _, _, confusion = evaluate(model, val_loader, device, num_classes, num_passes=5, return_confusion=True)
        print_confusion_matrix(confusion, class_names)
        print("\nRead this as: for each TRUE class (row), where did the model's predictions actually land?")
        print("A class that's confused with another shows up as a large off-diagonal number in that row.")

    torch.save({
        "model_state_dict": final_state,
        "class_names": class_names,
        "points_per_sample": args.points_per_sample,
        "architecture": args.architecture,
    }, args.output)
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to .npz produced by prepare_dataset.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--points_per_sample", type=int, default=1024,
                         help="Number of points fed to the model per training example")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--class_weighted_loss", action="store_true",
                         help="Upweight underrepresented/harder classes (e.g. Screw) in the loss function")
    parser.add_argument("--architecture", choices=["pointnet", "pointnet2"], default="pointnet",
                         help="pointnet2 adds local neighborhood features via hierarchical set "
                              "abstraction -- slower per epoch but better suited to small/scattered "
                              "components like Screw")
    parser.add_argument("--output", default="pointnet_seg_model.pt")
    args = parser.parse_args()

    train(args)
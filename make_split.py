"""
Split points_dataset_proportional.npz into a train set and a held-out
test set, by phone name. Held-out phones are fully excluded from
training (never touch train.py's internal val split either).

Usage:
    # Explicitly name the 5 phones to hold out
    python make_split.py --dataset points_dataset_proportional.npz \
        --holdout phone3 phone10 phone17 phone22 phone29

    # Or just ask for N random held-out phones (fixed seed = reproducible)
    python make_split.py --dataset points_dataset_proportional.npz \
        --n_holdout 5 --seed 42

Outputs:
    train_split.npz   -> feed this into train.py as --dataset
    test_split.npz    -> feed this into evaluate_holdout.py
"""

import argparse
import numpy as np

CLASS_NAMES_DEFAULT = ["Battery", "Camera", "Screw", "Other"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to full proportional .npz")
    parser.add_argument("--holdout", nargs="*", default=None,
                         help="Explicit phone names to hold out (must match phone_names in npz)")
    parser.add_argument("--n_holdout", type=int, default=None,
                         help="If --holdout not given, randomly pick this many phones to hold out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_all_classes", action="store_true",
                         help="Only consider phones that contain all classes as holdout candidates "
                              "(useful since some phones are missing component classes)")
    parser.add_argument("--train_out", default="train_split.npz")
    parser.add_argument("--test_out", default="test_split.npz")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    points = list(data["points"])
    labels = list(data["labels"])
    phone_names = list(data["phone_names"])
    class_names = list(data["class_names"])
    num_classes = len(class_names)

    name_to_idx = {name: i for i, name in enumerate(phone_names)}

    if args.holdout:
        missing = [n for n in args.holdout if n not in name_to_idx]
        if missing:
            raise ValueError(f"These phone names aren't in the dataset: {missing}\n"
                              f"Available: {phone_names}")
        holdout_names = args.holdout
    else:
        n_holdout = args.n_holdout or 5
        candidates = phone_names
        if args.require_all_classes:
            candidates = []
            for name in phone_names:
                idx = name_to_idx[name]
                present = set(np.unique(labels[idx]).tolist())
                if present == set(range(num_classes)):
                    candidates.append(name)
            print(f"{len(candidates)}/{len(phone_names)} phones contain all {num_classes} classes; "
                  f"sampling holdout from those.")
            if len(candidates) < n_holdout:
                raise ValueError(f"Only {len(candidates)} phones have all classes present, "
                                  f"can't hold out {n_holdout}. Lower --n_holdout or drop "
                                  f"--require_all_classes.")

        rng = np.random.default_rng(args.seed)
        holdout_names = list(rng.choice(candidates, size=n_holdout, replace=False))

    holdout_set = set(holdout_names)
    train_idx = [i for i, n in enumerate(phone_names) if n not in holdout_set]
    test_idx = [i for i, n in enumerate(phone_names) if n in holdout_set]

    def subset(idx_list):
        return (
            np.array([points[i] for i in idx_list], dtype=object),
            np.array([labels[i] for i in idx_list], dtype=object),
            np.array([phone_names[i] for i in idx_list]),
        )

    train_pts, train_lbls, train_names = subset(train_idx)
    test_pts, test_lbls, test_names = subset(test_idx)

    np.savez(args.train_out, points=train_pts, labels=train_lbls,
             phone_names=train_names, class_names=np.array(class_names))
    np.savez(args.test_out, points=test_pts, labels=test_lbls,
             phone_names=test_names, class_names=np.array(class_names))

    print(f"\nHeld out ({len(test_names)}): {list(test_names)}")
    print(f"Train set ({len(train_names)}): {list(train_names)}")

    # Sanity: report per-class point counts in the held-out set
    print("\nHeld-out set class composition:")
    for i, name in enumerate(test_names):
        idx = test_idx[i]
        counts = {class_names[c]: int((labels[idx] == c).sum()) for c in range(num_classes)}
        missing_classes = [c for c, cnt in counts.items() if cnt == 0]
        flag = f"  <- MISSING: {missing_classes}" if missing_classes else ""
        print(f"  {name}: {counts}{flag}")

    print(f"\nSaved: {args.train_out}, {args.test_out}")


if __name__ == "__main__":
    main()

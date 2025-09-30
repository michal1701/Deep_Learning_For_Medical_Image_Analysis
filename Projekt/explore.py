# dentex_explore.py
from pathlib import Path
import json
from collections import defaultdict

BASE = Path("DENTEX")

TRAIN_ROOT = BASE / "training_data" / "training_data"
VAL_ROOT   = BASE / "validation_data" / "validation_data"
TEST_ROOT  = BASE / "test_data"

TRAIN_SUBSETS = {
    "quadrant_enumeration_disease": TRAIN_ROOT / "quadrant-enumeration-disease",
    "quadrant_enumeration":         TRAIN_ROOT / "quadrant_enumeration",
    "quadrant":                     TRAIN_ROOT / "quadrant",
    "unlabelled":                   TRAIN_ROOT / "unlabelled",
}

def count_images(folder: Path):
    return sum(1 for _ in (folder.rglob("*.png"))) + sum(1 for _ in (folder.rglob("*.jpg")))

def main():
    print("=== DENTEX — Dataset Summary (tailored to your tree) ===\n")

    # -------- TRAIN --------
    print("[Train]")
    for name, root in TRAIN_SUBSETS.items():
        xrays = root / "xrays" if (root / "xrays").exists() else root  # unlabelled/xrays exists
        n = count_images(xrays)
        ann_jsons = list(root.glob("*.json"))
        print(f"  - {name:28s}  images: {n:4d}   jsons: {len(ann_jsons)} ({[p.name for p in ann_jsons]})")
    print()

    # -------- VAL --------
    print("[Validation]")
    # from your tree: validation_data/validation_data/quadrant_enumeration_disease/xrays/*.png
    val_qed = VAL_ROOT / "quadrant_enumeration_disease"
    n_val = count_images(val_qed / "xrays")
    print(f"  - quadrant_enumeration_disease  images: {n_val:4d}")
    print()

    # -------- TEST --------
    print("[Test]")
    test_in  = TEST_ROOT / "disease" / "input"
    test_lab = TEST_ROOT / "disease" / "label"
    n_test_img = count_images(test_in)
    n_test_lab = len(list(test_lab.glob("*.json")))
    print(f"  - disease/input   images: {n_test_img:4d}")
    print(f"  - disease/label   jsons : {n_test_lab:4d}")
    print()

    # -------- File-type histogram (like you printed) --------
    suffix_hist = defaultdict(int)
    for p in BASE.rglob("*"):
        if p.is_file():
            suffix_hist[p.suffix] += 1
    print("[File extensions in DENTEX/]")
    for k, v in sorted(suffix_hist.items(), key=lambda x: (-x[1], x[0] or "<none>")):
        print(f"  {k or '<none>'}: {v}")
    print()

if __name__ == "__main__":
    main()

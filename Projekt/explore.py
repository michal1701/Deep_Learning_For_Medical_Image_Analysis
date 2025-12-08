import os
import json
import random
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# === AUTO-DETECT NESTED ROOT ===
ROOT = Path("data/DENTEX")
subfolder = ROOT / "DENTEX"
if subfolder.exists():
    DATA_DIR = subfolder
else:
    DATA_DIR = ROOT

TRAIN_DIR = DATA_DIR / "training_data"
VAL_DIR = DATA_DIR / "validation_data"
DISEASE_DIR = DATA_DIR / "disease"
VAL_FILE = DATA_DIR / "validation_triple.json"

print(f"📂 Using dataset base: {DATA_DIR}\n")

# === 1. DIRECTORY SUMMARY ===
def explore_structure(base_dir):
    print("📁 Directory structure:\n")
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(str(base_dir), "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}- {os.path.basename(root)}/ ({len(files)} files, {len(dirs)} dirs)")
    print("\n✅ Done.\n")

# === 2. SHOW SAMPLE IMAGES ===
def show_samples(img_dir, n=6, title=None):
    paths = list(Path(img_dir).rglob("*.jpg")) + list(Path(img_dir).rglob("*.png"))
    if not paths:
        print(f"⚠️ No images found in {img_dir}")
        return
    samples = random.sample(paths, min(n, len(paths)))
    plt.figure(figsize=(12, 6))
    for i, p in enumerate(samples):
        img = Image.open(p)
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(p.stem[:15])
        plt.axis("off")
    plt.suptitle(title or img_dir.name)
    plt.tight_layout()
    plt.show()

# === 3. EXPLORE DISEASE LABELS ===
def explore_disease_pairs(disease_dir):
    input_dir = disease_dir / "input"
    label_dir = disease_dir / "label"
    inputs = sorted(list(input_dir.glob("*.jpg")))
    labels = sorted(list(label_dir.glob("*.jpg")))
    print(f"🧬 Disease samples: {len(inputs)} inputs, {len(labels)} labels.")
    if inputs and labels:
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for i in range(4):
            if i < len(inputs):
                axes[0, i].imshow(Image.open(inputs[i]), cmap='gray')
                axes[0, i].set_title("Input")
                axes[0, i].axis("off")
            if i < len(labels):
                axes[1, i].imshow(Image.open(labels[i]), cmap='gray')
                axes[1, i].set_title("Label")
                axes[1, i].axis("off")
        plt.tight_layout()
        plt.show()

# === 4. VALIDATION JSON SUMMARY ===
def explore_validation_json(val_file):
    if not val_file.exists():
        print("⚠️ Validation JSON not found.\n")
        return
    with open(val_file, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded validation JSON with {len(data)} entries.")
    print("Example entry:\n", json.dumps(data[0], indent=2) if isinstance(data, list) else data)

# === MAIN ===
if __name__ == "__main__":
    explore_structure(DATA_DIR)

    # Display samples from each dataset type
    for subset in ["quadrant", "quadrant_enumeration", "quadrant-enumeration-disease", "unlabelled"]:
        xr_dir = TRAIN_DIR / subset / "xrays"
        if xr_dir.exists():
            show_samples(xr_dir, title=subset)

    # Disease input vs label visualization
    if DISEASE_DIR.exists():
        explore_disease_pairs(DISEASE_DIR)

    # Validation samples (if present)
    val_xrays = VAL_DIR / "quadrant_enumeration_disease" / "xrays"
    if val_xrays.exists():
        show_samples(val_xrays, title="Validation X-rays")

    explore_validation_json(VAL_FILE)
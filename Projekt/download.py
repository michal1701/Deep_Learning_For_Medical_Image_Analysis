# 3) download_dentex.py
from huggingface_hub import snapshot_download
from pathlib import Path

# Hugging Face dataset repo id
REPO_ID = "ibrahimhamamci/DENTEX"   # DENTEX dataset
TARGET_DIR = Path("data/DENTEX")    # change if you want a different path

# Download the full dataset snapshot (≈12 GB)
local_dir = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(TARGET_DIR),
    local_dir_use_symlinks=False,   # set True if you prefer symlinks
    resume_download=True,           # safe to resume if interrupted
    max_workers=8                   # parallelism for speed
)

print(f"✅ Downloaded DENTEX to: {local_dir}")

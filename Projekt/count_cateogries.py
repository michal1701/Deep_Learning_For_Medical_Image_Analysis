# dentex_stats_fixed.py
from pathlib import Path
import json
from collections import Counter

json_file = Path("DENTEX/training_data/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json")

with open(json_file) as f:
    data = json.load(f)

# Extract mappings
cat1_map = {c["id"]: c["name"] for c in data["categories_1"]}
cat2_map = {c["id"]: c["name"] for c in data["categories_2"]}
cat3_map = {c["id"]: c["name"] for c in data["categories_3"]}

# Extract annotation list
annotations = data["annotations"]

# Count category frequencies
c1 = Counter([cat1_map[a["category_id_1"]] for a in annotations])
c2 = Counter([cat2_map[a["category_id_2"]] for a in annotations])
c3 = Counter([cat3_map[a["category_id_3"]] for a in annotations])

print("=== QUADRANT (categories_1) ===")
for k, v in c1.most_common():
    print(f"{k:15s} : {v}")

print("\n=== TOOTH (categories_2) ===")
for k, v in c2.most_common():
    print(f"{k:15s} : {v}")

print("\n=== DIAGNOSIS (categories_3) ===")
for k, v in c3.most_common():
    print(f"{k:15s} : {v}")

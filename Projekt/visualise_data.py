# dentex_visualize_robust.py
from pathlib import Path
import json, random
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE = Path("DENTEX")
TRAIN = {
    "qed": BASE/"training_data"/"training_data"/"quadrant-enumeration-disease",
    "qe":  BASE/"training_data"/"training_data"/"quadrant_enumeration",
    "q":   BASE/"training_data"/"training_data"/"quadrant",
    "unl": BASE/"training_data"/"training_data"/"unlabelled",
}
VAL_XRAYS  = BASE/"validation_data"/"validation_data"/"quadrant_enumeration_disease"/"xrays"
TEST_IN    = BASE/"test_data"/"disease"/"input"
TEST_LABEL = BASE/"test_data"/"disease"/"label"

def list_imgs(p: Path):
    return sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")))

# ---------- JSON helpers (filename, objects, fields) ----------
def get_filename(rec: Any) -> str:
    if isinstance(rec, dict):
        for k in ("filename","file_name","image","image_name","name","img","path"):
            if k in rec: return str(rec[k])
    return ""

def get_objects(rec: Any) -> List[Dict[str,Any]]:
    if isinstance(rec, dict):
        for k in ("objects","annotations","labels","boxes","bboxes","detections"):
            if k in rec and isinstance(rec[k], list): return rec[k]
        # sometimes record itself is an object:
        if "bbox" in rec or "box" in rec: return [rec]
    if isinstance(rec, list): return rec
    return []

def get_label(obj: Dict[str,Any]) -> str:
    # compose a readable label if we have parts (quadrant/tooth/diagnosis)
    parts = []
    for k in ("label","class","category","diagnosis","disease","tooth","tooth_id","quadrant"):
        if k in obj and obj[k] not in (None,""):
            parts.append(str(obj[k]))
    return "/".join(parts) if parts else ""

def bbox_from_xywh(b: List[float]) -> Tuple[float,float,float,float]:
    x,y,w,h = b[:4]; return float(x),float(y),float(w),float(h)

def bbox_from_x1y1x2y2(b: List[float]) -> Tuple[float,float,float,float]:
    x1,y1,x2,y2 = b[:4]; return float(x1),float(y1),float(x2-x1),float(y2-y1)

def try_bbox(obj: Dict[str,Any], W: Optional[int]=None, H: Optional[int]=None) -> Optional[Tuple[float,float,float,float]]:
    # 1) common containers
    cand = None
    for key in ("bbox","box","rectangle","rect"):
        if key in obj and isinstance(obj[key], (list,tuple)) and len(obj[key])>=4:
            cand = list(map(float, obj[key][:4])); break
    if cand is not None:
        # guess format: if x2>x1 and y2>y1 and both >1, treat as x1y1x2y2
        x,y,w,h = cand[:4]
        if x<0 or y<0:  # maybe normalized yolo (cx,cy,w,h in [0,1])
            pass
        # heuristics
        if (w>1 and h>1 and x<w and y<h):  # ambiguous, still assume xywh
            return bbox_from_xywh(cand)
        # if looks like x2,y2
        if w>1 and h>1 and w> x and h> y:
            # detect likely x1y1x2y2 (very common in academic JSONs)
            return bbox_from_x1y1x2y2(cand)
        return bbox_from_xywh(cand)

    # 2) YOLO normalized (cx,cy,w,h) -> need image size
    if all(k in obj for k in ("cx","cy","w","h")) and W and H:
        cx,cy,w,h = float(obj["cx"]),float(obj["cy"]),float(obj["w"]),float(obj["h"])
        if 0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1:
            x = (cx - w/2)*W; y = (cy - h/2)*H
            return x,y,w*W,h*H

    return None

# ---------- Loaders ----------
def load_train_mapping(folder: Path) -> Dict[str, List[Dict[str,Any]]]:
    """Map image stem -> list of object dicts for train subsets (single JSON file)."""
    jfiles = list(folder.glob("*.json"))
    if not jfiles: return {}
    data = json.loads(jfiles[0].read_text())
    mapping: Dict[str, List[Dict[str,Any]]] = {}
    if isinstance(data, dict):
        for k,v in data.items():
            stem = Path(k).stem
            mapping[stem] = get_objects(v) if isinstance(v, dict) else (v if isinstance(v, list) else [])
    elif isinstance(data, list):
        for rec in data:
            fname = get_filename(rec)
            stem = Path(fname).stem if fname else None
            if stem:
                mapping.setdefault(stem, []).extend(get_objects(rec) or [])
    return mapping

def load_test_ann(stem: str) -> List[Dict[str,Any]]:
    jf = TEST_LABEL/f"{stem}.json"
    if jf.exists():
        data = json.loads(jf.read_text())
        return get_objects(data) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    return []

# ---------- Viz ----------
def show_with_boxes(img_path: Path, objs: List[Dict[str,Any]]):
    im = Image.open(img_path).convert("RGB")
    W,H = im.size
    fig,ax = plt.subplots(figsize=(10,10))
    ax.imshow(im)
    drawn=0
    for o in objs:
        bb = try_bbox(o, W, H)
        if not bb: continue
        x,y,w,h = bb
        ax.add_patch(patches.Rectangle((x,y), w,h, fill=False, linewidth=2))
        lab = get_label(o)
        if lab:
            ax.text(x, max(0,y-4), lab, fontsize=8, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        drawn += 1
    ax.set_title(f"{img_path.name} — boxes: {drawn}")
    ax.axis("off")
    plt.show()

def visualize(split="train_qed", n=6):
    if split=="train_qed":
        root = TRAIN["qed"]; imgs = list_imgs(root/"xrays"); ann = load_train_mapping(root)
        for p in random.sample(imgs, min(n, len(imgs))): show_with_boxes(p, ann.get(p.stem, []))
    elif split=="train_qe":
        root = TRAIN["qe"];  imgs = list_imgs(root/"xrays"); ann = load_train_mapping(root)
        for p in random.sample(imgs, min(n, len(imgs))): show_with_boxes(p, ann.get(p.stem, []))
    elif split=="train_q":
        root = TRAIN["q"];   imgs = list_imgs(root/"xrays"); ann = load_train_mapping(root)
        for p in random.sample(imgs, min(n, len(imgs))): show_with_boxes(p, ann.get(p.stem, []))
    elif split=="train_unl":
        imgs = list_imgs(TRAIN["unl"]/ "xrays")
        for p in random.sample(imgs, min(n, len(imgs))): show_with_boxes(p, [])
    elif split=="val_qed":
        imgs = list_imgs(VAL_XRAYS)
        for p in random.sample(imgs, min(n, len(imgs))): show_with_boxes(p, [])
    elif split=="test":
        imgs = list_imgs(TEST_IN)
        for p in random.sample(imgs, min(n, len(imgs))): show_with_boxes(p, load_test_ann(p.stem))
    else:
        raise ValueError("split ∈ {'train_qed','train_qe','train_q','train_unl','val_qed','test'}")

if __name__ == "__main__":
    visualize("train_qed", 6)

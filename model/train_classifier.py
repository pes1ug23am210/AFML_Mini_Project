# ------------------------------------------------------------
#  image.py  –  EfficientNet‑B0 classifier
#  Works with your current data.yaml (../train/images, path: D:\mod)
#  No manual edits required – paths are fixed automatically.
# ------------------------------------------------------------
import os
import yaml
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------- CONFIG ----------------------
YAML_PATH   = "data.yaml"
BATCH_SIZE  = 32
EPOCHS      = 60
LR          = 1e-3
PATIENCE    = 12
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TMP_ROOT    = Path("./tmp_classifier")
# ------------------------------------------------------------

# ---------------------- 1. LOAD YAML & FIX PATHS ----------------------
yaml_path = Path(YAML_PATH)
if not yaml_path.exists():
    raise FileNotFoundError(f"{YAML_PATH} not found in {os.getcwd()}")

with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

# Root from `path:` (D:\mod)
root_dir = Path(cfg["path"]).resolve()
print(f"Dataset root (from path:): {root_dir}")

# Resolve relative splits (../train/images → D:\train\images)
def resolve_split(key):
    rel = cfg.get(key)
    if not rel:
        return None
    candidate = (root_dir / ".." / rel).resolve()
    # ---- AUTO‑CORRECT if the folder does NOT exist ----
    if not candidate.exists():
        # Try inside the root (./train/images)
        alt = (root_dir / rel.replace("../", "./")).resolve()
        if alt.exists():
            print(f"Warning: {candidate} not found → using {alt}")
            return str(alt)
        else:
            raise FileNotFoundError(f"Neither {candidate} nor {alt} exists")
    return str(candidate)

train_img = resolve_split("train")
val_img   = resolve_split("val")
test_img  = resolve_split("test")

class_names = list(cfg["names"].values())
num_classes = len(class_names)

print(f"Classes ({num_classes}): {class_names}")
print(f"Train images : {train_img}")
print(f"Val   images : {val_img}")
if test_img:
    print(f"Test  images : {test_img}")

# ---------------------- 2. BUILD CLASS FOLDERS ----------------------
def _make_split(img_dir: str, lbl_dir: str, out_root: Path):
    img_p = Path(img_dir)
    lbl_p = Path(lbl_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not img_p.exists():
        raise FileNotFoundError(f"Image folder not found: {img_p}")
    if not lbl_p.exists():
        raise FileNotFoundError(f"Label folder not found: {lbl_p}")

    copied = 0
    for lbl_file in lbl_p.glob("*.txt"):
        stem = lbl_file.stem
        img_files = list(img_p.glob(f"{stem}.*"))
        if not img_files:
            continue
        img_file = img_files[0]

        try:
            with open(lbl_file, "r") as f:
                line = f.readline().strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                if cls_id >= num_classes:
                    continue
        except Exception:
            continue

        cls_name = class_names[cls_id]
        dest_dir = out_root / cls_name
        dest_dir.mkdir(exist_ok=True)
        dest_path = dest_dir / img_file.name
        if not dest_path.exists():
            shutil.copy(img_file, dest_path)
            copied += 1

    print(f"  → Copied {copied} images to {out_root}")
    if copied == 0:
        raise RuntimeError(f"No images copied in {out_root}")

def build_split(name: str):
    img_dir = resolve_split(name)
    lbl_dir = img_dir.replace("images", "labels")
    tmp_dir = TMP_ROOT / name
    print(f"Building {name} split → {tmp_dir}")
    _make_split(img_dir, lbl_dir, tmp_dir)
    return str(tmp_dir)

# Build splits
train_dir = build_split("train")
val_dir   = build_split("val")
test_dir  = build_split("test") if cfg.get("test") else None

# ---------------------- 3. DATA LOADERS ----------------------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(train_dir, transform=train_tf)
val_set   = datasets.ImageFolder(val_dir,   transform=val_tf)
test_set  = datasets.ImageFolder(test_dir,  transform=val_tf) if test_dir else None

print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True) if test_set else None

# ---------------------- 4. MODEL ----------------------
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, num_classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------- 5. TRAINING HELPERS ----------------------
def train_one_epoch():
    model.train()
    loss_sum = correct = total = 0.0
    for x, y in tqdm(train_loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        _, pred = out.max(1)
        total   += y.size(0)
        correct += pred.eq(y).sum().item()
    return loss_sum / len(train_loader), 100.0 * correct / total

def evaluate(loader, desc):
    model.eval()
    loss_sum = correct = total = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item()
            _, pred = out.max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
    return loss_sum / len(loader), 100.0 * correct / total, preds, trues

# ---------------------- 6. MAIN TRAINING LOOP ----------------------
best_val = 0.0
patience_cnt = 0
hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print(f"\nStarting training on {DEVICE} …\n")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch()
    val_loss, val_acc, _, _ = evaluate(val_loader, "Val")
    scheduler.step()

    hist["train_loss"].append(tr_loss)
    hist["train_acc"].append(tr_acc)
    hist["val_loss"].append(val_loss)
    hist["val_acc"].append(val_acc)

    print(f"Epoch {epoch:02d} | Train L:{tr_loss:.4f} A:{tr_acc:5.2f}% | Val L:{val_loss:.4f} A:{val_acc:5.2f}%")

    if val_acc > best_val:
        best_val = val_acc
        patience_cnt = 0
        torch.save(model.state_dict(), "classifier_best.pth")
        print("  → BEST model saved")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ---------------------- 7. TEST (optional) ----------------------
model.load_state_dict(torch.load("classifier_best.pth"))

if test_loader:
    _, test_acc, t_pred, t_true = evaluate(test_loader, "Test")
    print(f"\nTest accuracy: {test_acc:.2f}%")
    print("\nClassification report:")
    print(classification_report(t_true, t_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(t_true, t_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test)')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
else:
    print("\nNo test split → skipping final test.")

# ---------------------- 8. PLOTS ----------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist["train_loss"], label="Train")
plt.plot(hist["val_loss"],   label="Val")
plt.legend(); plt.title("Loss")
plt.subplot(1,2,2)
plt.plot(hist["train_acc"], label="Train")
plt.plot(hist["val_acc"],   label="Val")
plt.legend(); plt.title("Accuracy")
plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# ---------------------- 9. ONNX EXPORT ----------------------
model.eval()
dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
torch.onnx.export(
    model, dummy, "classifier.onnx",
    export_params=True, opset_version=17,
    do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print("\nONNX model saved → classifier.onnx")

# ---------------------- 10. CLEANUP ----------------------
shutil.rmtree(TMP_ROOT, ignore_errors=True)
print(f"\nTraining finished! Best validation accuracy: {best_val:.2f}%")
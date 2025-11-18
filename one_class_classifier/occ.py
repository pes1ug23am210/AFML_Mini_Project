# oneclass_model.py: Independent One-Class Detection with Deep SVDD (Load Existing + Improved Conf)
# Usage: 
# 1. (Optional) Place 20 training images in ./support_pics/ (only if no model exists).
# 2. Place test images in ./pics/.
# 3. Run: python oneclass_model.py
# - If 'oneclass_model.pth' exists: Loads and tests instantly.
# - Else: Trains from support_pics/ and saves model.
# Outputs: Binary predictions on test images.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

# Config
SUPPORT_DIR = "./support_pics"  # Your 20 training images here (if training)
TEST_DIR = "./pics"            # Test images here
MODEL_PATH = 'oneclass_model.pth'
BATCH_SIZE = 8
PRE_EPOCHS = 50
SVDD_EPOCHS = 100
LR = 1e-4
NU = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset for support (positives only)
class SupportDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = [f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.root = root
        self.transform = transform
        print(f"Found {len(self.images)} support images in {root}")
    
    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# Test Dataset
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = [f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.root = root
        self.transform = transform
        print(f"Found {len(self.images)} test images in {root}")
    
    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.images[idx]

# Model: EfficientNet-based SVDD
class SVDDNet(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Extract 1280-dim features
        # Freeze early layers for few-shot
        for param in self.backbone.features[:4].parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1280, latent_dim)
        self.center = nn.Parameter(torch.zeros(1, latent_dim))
    
    def forward(self, x):
        feats = self.backbone(x)
        z = self.fc(feats)
        dist = torch.sum((z - self.center) ** 2, dim=1)
        return dist, z

# AutoEncoder for pretraining
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        backbone = models.efficientnet_b0(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1280),
            nn.ReLU(),
            nn.Linear(1280, 224*224*3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z).view(x.size(0), 3, 224, 224)
        return recon, z

# SVDD Loss
def svdd_loss(dist, nu=NU):
    radius = dist.mean()
    return radius + (nu / len(dist)) * torch.sum(torch.max(dist - radius, torch.zeros_like(dist)))

# Standalone inference function
def predict_single(image_path, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first or provide path.")
    ckpt = torch.load(model_path, map_location='cpu')
    net = SVDDNet()
    net.load_state_dict(ckpt['model_state'])
    net.center.data = ckpt['center']
    net.eval()
    
    transform_single = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("RGB")
    tensor = transform_single(img).unsqueeze(0)
    
    with torch.no_grad():
        dist, _ = net(tensor)
        thresh = ckpt['threshold']
        is_match = dist.item() < thresh
        # IMPROVED CONF: For matches: 1 - (dist/thresh); For non: exp(-(dist-thresh)/thresh) clamped [0,1]
        if is_match:
            conf = 1 - (dist.item() / thresh)
        else:
            conf = np.exp(- (dist.item() - thresh) / thresh)
        conf = max(0, min(1, conf))  # Clamp
        return is_match, conf

# Main
if __name__ == "__main__":
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Create {TEST_DIR} for test images.")
    
    # Check for existing model
    if os.path.exists(MODEL_PATH):
        print(f"✅ Existing model found at {MODEL_PATH}. Loading for inference only...")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        svdd_net = SVDDNet().to(DEVICE)
        svdd_net.load_state_dict(ckpt['model_state'])
        svdd_net.center.data = ckpt['center']
        svdd_net.eval()
        threshold = ckpt['threshold']
        print(f"Threshold loaded: {threshold:.4f}")
    else:
        print("❌ No model found. Training new one...")
        if not os.path.exists(SUPPORT_DIR):
            raise FileNotFoundError(f"Create {SUPPORT_DIR} with 20 images of your target object.")
        
        # Data
        support_ds = SupportDataset(SUPPORT_DIR, transform)
        support_dl = DataLoader(support_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # Pretrain AE
        ae = AutoEncoder().to(DEVICE)
        ae_opt = optim.Adam(ae.parameters(), lr=LR)
        ae_crit = nn.MSELoss()
        
        print("Pretraining AutoEncoder...")
        ae.train()
        for epoch in tqdm(range(PRE_EPOCHS), desc="AE"):
            total_loss = 0
            for batch in support_dl:
                batch = batch.to(DEVICE)
                ae_opt.zero_grad()
                recon, _ = ae(batch)
                loss = ae_crit(recon, batch)
                loss.backward()
                ae_opt.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"AE Epoch {epoch}: Avg Loss {total_loss / len(support_dl):.4f}")
        
        # Init SVDD
        svdd_net = SVDDNet().to(DEVICE)
        svdd_net.fc.weight.data = ae.encoder[-1].weight.data
        svdd_net.fc.bias.data = ae.encoder[-1].bias.data
        
        # SVDD Training
        svdd_opt = optim.Adam(svdd_net.parameters(), lr=LR)
        
        print("Training SVDD...")
        svdd_net.train()
        for epoch in tqdm(range(SVDD_EPOCHS), desc="SVDD"):
            total_loss = 0
            for batch in support_dl:
                batch = batch.to(DEVICE)
                svdd_opt.zero_grad()
                dist, _ = svdd_net(batch)
                loss = svdd_loss(dist)
                loss.backward()
                svdd_opt.step()
                total_loss += loss.item()
            if epoch % 20 == 0:
                print(f"SVDD Epoch {epoch}: Avg Loss {total_loss / len(support_dl):.4f}")
        
        # Compute threshold
        svdd_net.eval()
        support_dists = []
        with torch.no_grad():
            for batch in support_dl:
                batch = batch.to(DEVICE)
                dist, _ = svdd_net(batch)
                support_dists.extend(dist.cpu().numpy())
        threshold = np.mean(support_dists) + 2 * np.std(support_dists)
        print(f"Threshold set to {threshold:.4f}")
        
        # Save
        torch.save({
            'model_state': svdd_net.state_dict(),
            'center': svdd_net.center.data,
            'threshold': threshold
        }, MODEL_PATH)
        print(f"Model saved as '{MODEL_PATH}'")
    
    # Inference on pics
    test_ds = TestDataset(TEST_DIR, transform)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for imgs, filenames in test_dl:
            imgs = imgs.to(DEVICE)
            dists, _ = svdd_net(imgs)
            for dist, fname in zip(dists.cpu().numpy(), filenames):
                is_match = dist < threshold
                # IMPROVED CONF
                if is_match:
                    conf = 1 - (dist / threshold)
                else:
                    conf = np.exp(- (dist - threshold) / threshold)
                conf = max(0, min(1, conf))
                predictions.append({
                    'filename': fname,
                    'distance': dist,
                    'is_match': is_match,
                    'confidence': conf
                })
    
    # Print results
    print("\nTest Results:")
    print("-" * 50)
    matches = sum(1 for p in predictions if p['is_match'])
    print(f"Matches: {matches}/{len(predictions)} ({matches/len(predictions)*100:.1f}%)")
    
    for p in predictions:
        status = "✅ YES" if p['is_match'] else "❌ NO"
        print(f"{p['filename']}: {status} (conf: {p['confidence']:.2f}, dist: {p['distance']:.3f})")
    
    print(f"\nStandalone predict_single() function available for new images (see code above)!")
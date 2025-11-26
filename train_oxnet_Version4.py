import argparse
from pathlib import Path
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class PNGFolder(Dataset):
    def __init__(self, root, classes=None, target_size=None, transform=None):
        self.root = Path(root)
        self.transform = transform
        if classes is None:
            classes = tuple(
                sorted(
                    [p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob("*.png"))]
                )
            )
        else:
            classes = tuple(classes)
        if not classes:
            raise ValueError(f"No class folders with PNG files were found in {self.root}")
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for c in self.classes:
            cls_dir = self.root / c
            if not cls_dir.exists():
                raise FileNotFoundError(f"Class folder {cls_dir} does not exist")
            files = sorted(cls_dir.glob("*.png"))
            if not files:
                raise FileNotFoundError(f"No PNG files found under {cls_dir}")
            for p in files:
                self.samples.append((p, self.class_to_idx[c]))

        if not self.samples:
            raise ValueError(f"Dataset at {self.root} is empty")

        with Image.open(self.samples[0][0]) as probe:
            self.native_size = probe.size  # (width, height)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = self.native_size
        self.image_size = self.target_size
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("L")
        
        if self.transform:
            x = self.transform(img)
        else:
            if self.target_size is not None and img.size != self.target_size:
                img = img.resize(self.target_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W], 0..1
            # add channel dimension
            x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
            
        return x, torch.tensor(y, dtype=torch.long)

class OXNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28->28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 28->14
            
            nn.Conv2d(32, 64, 3, padding=1), # 14->14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 14->7
            
            nn.Conv2d(64, 64, 3, padding=1), # 7->7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 7->3 (floor) -> actually 3x3 if input is 28
        )
        # 28 -> 14 -> 7 -> 3.  3*3*64 = 576
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, opt, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_acc = 0.0
    running_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
        running_acc += accuracy(logits.detach(), y) * x.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_acc = 0.0
    running_loss = 0.0
    for x, y in tqdm(loader, desc="val  ", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        running_acc += accuracy(logits, y) * x.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

def export_onnx(model, onnx_path: Path, device, input_size=(28, 28)):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 1, input_size[1], input_size[0], dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes=None,
    )
    print(f"Exported ONNX model to {onnx_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset", help="dataset root dir with train/ and val/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--onnx", type=str, default="model/model.onnx")
    parser.add_argument("--img-size", type=int, default=28, help="Image size to which samples are resized")
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Comma-separated class order (defaults to auto-detect from train folder)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explicit_classes = tuple(c.strip() for c in args.classes.split(",") if c.strip()) or None
    target_hw = (args.img_size, args.img_size)

    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize(target_hw),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(target_hw),
        transforms.ToTensor(),
    ])

    train_ds = PNGFolder(Path(args.data) / "train", classes=explicit_classes, target_size=target_hw, transform=train_transform)
    val_ds = PNGFolder(Path(args.data) / "val", classes=train_ds.classes, target_size=target_hw, transform=val_transform)
    print(f"Classes: {train_ds.classes}")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = OXNet(num_classes=len(train_ds.classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_acc = eval_epoch(model, val_loader, device)
        scheduler.step()
        print(f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            # Save best model for export
            torch.save(model.state_dict(), "best_model.pth")

    # Load best model for export
    if Path("best_model.pth").exists():
        model.load_state_dict(torch.load("best_model.pth"))
        Path("best_model.pth").unlink() # Clean up

    export_onnx(model, Path(args.onnx), device, input_size=train_ds.image_size)

if __name__ == "__main__":
    main()
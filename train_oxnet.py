import argparse
from pathlib import Path
import random
import json
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PNGFolder(Dataset):
    def __init__(self, root, classes=("O", "X", "checkmark")):
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            cls_dir = Path(root) / c
            for p in sorted(cls_dir.glob("*.png")):
                self.samples.append((p, self.class_to_idx[c]))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W], 0..1
        # add channel dimension
        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return x, torch.tensor(y, dtype=torch.long)

class OXNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0),  # 28->26
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 26->13
            nn.Conv2d(16, 32, 3, padding=0), # 13->11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 11->5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # 3 classes: O, X, checkmark
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

def export_onnx(model, onnx_path: Path, device):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32, device=device)
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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--onnx", type=str, default="model/model.onnx")
    parser.add_argument("--results", type=str, default="results", help="directory to save result files")
    args = parser.parse_args()

    # Create results directory
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_ds = PNGFolder(Path(args.data) / "train")
    val_ds = PNGFolder(Path(args.data) / "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    model = OXNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Track training history
    training_history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_acc = eval_epoch(model, val_loader, device)
        
        # Record training history
        training_history["epochs"].append(epoch)
        training_history["train_loss"].append(tr_loss)
        training_history["train_acc"].append(tr_acc)
        training_history["val_loss"].append(va_loss)
        training_history["val_acc"].append(va_acc)
        
        print(f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc

    training_time = time.time() - start_time
    
    # Save training results
    training_results = {
        "dataset_info": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "classes": ["O", "X", "checkmark"],
            "class_mapping": {"O": 0, "X": 1, "checkmark": 2}
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "device": str(device)
        },
        "training_history": training_history,
        "final_results": {
            "best_validation_accuracy": best_acc,
            "final_train_loss": tr_loss,
            "final_train_acc": tr_acc,
            "final_val_loss": va_loss,
            "final_val_acc": va_acc,
            "training_time_seconds": training_time
        }
    }
    
    # Save results to JSON file
    with open(results_dir / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2, ensure_ascii=False)
    
    # Save training summary
    summary_text = f"""
ONNX-OX Training Results Summary
===============================

Dataset Information:
- Training samples: {len(train_ds)}
- Validation samples: {len(val_ds)}
- Classes: Circle (圈), Cross (叉), Checkmark (勾)

Training Configuration:
- Epochs: {args.epochs}
- Batch size: {args.batch}
- Learning rate: {args.lr}
- Device: {device}

Final Results:
- Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)
- Final training accuracy: {tr_acc:.4f} ({tr_acc*100:.2f}%)
- Final validation accuracy: {va_acc:.4f} ({va_acc*100:.2f}%)
- Training time: {training_time:.2f} seconds

Model Output:
- ONNX model saved to: {args.onnx}
- Results saved to: {results_dir}
"""
    
    with open(results_dir / "training_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"\nTraining completed successfully!")
    print(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Results saved to: {results_dir}")
    
    export_onnx(model, Path(args.onnx), device)

if __name__ == "__main__":
    main()
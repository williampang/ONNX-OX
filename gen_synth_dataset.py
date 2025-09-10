import argparse
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import numpy as np

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def add_noise(img: Image.Image, amount=0.02):
    # amount: percent of pixels to flip/add speckle
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape
    n = int(h * w * amount)
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    arr[ys, xs] = np.random.randint(0, 256, n)
    return Image.fromarray(arr, mode="L")

def draw_O(img_size: int, small_variant: bool = False) -> Image.Image:
    canvas = Image.new("L", (img_size, img_size), color=0)  # black bg
    draw = ImageDraw.Draw(canvas)
    
    # For small variants, use smaller margins and size
    if small_variant:
        margin = random.randint(6, 10)  # Larger margins for smaller shapes
        thickness = random.randint(1, 3)  # Thinner lines
    else:
        margin = random.randint(2, 6)
        thickness = random.randint(2, 5)
    
    x0 = margin + random.randint(0, 2)
    y0 = margin + random.randint(0, 2)
    x1 = img_size - margin - random.randint(0, 2)
    y1 = img_size - margin - random.randint(0, 2)
    
    # Draw outer ellipse by stroking multiple widths
    for t in range(thickness):
        draw.ellipse((x0 - t, y0 - t, x1 + t, y1 + t), outline=255)
    # Random rotation (draw on bigger canvas then rotate)
    angle = random.uniform(-20, 20)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    return canvas

def draw_X(img_size: int, small_variant: bool = False) -> Image.Image:
    canvas = Image.new("L", (img_size, img_size), color=0)  # black bg
    draw = ImageDraw.Draw(canvas)
    
    # For small variants, use smaller margins and size
    if small_variant:
        margin = random.randint(6, 10)  # Larger margins for smaller shapes
        thickness = random.randint(1, 3)  # Thinner lines
    else:
        margin = random.randint(2, 6)
        thickness = random.randint(2, 5)
    
    jitter = random.randint(-2, 2)
    # Two crossing lines
    for t in range(thickness):
        draw.line(
            (margin - t, margin + jitter, img_size - margin + t, img_size - margin + jitter),
            fill=255,
            width=1,
        )
        draw.line(
            (img_size - margin + t, margin + jitter, margin - t, img_size - margin + jitter),
            fill=255,
            width=1,
        )
    angle = random.uniform(-20, 20)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    return canvas

def draw_checkmark(img_size: int, small_variant: bool = False) -> Image.Image:
    canvas = Image.new("L", (img_size, img_size), color=0)  # black bg
    draw = ImageDraw.Draw(canvas)
    
    # For small variants, use smaller margins and size
    if small_variant:
        margin = random.randint(6, 10)  # Larger margins for smaller shapes
        thickness = random.randint(1, 3)  # Thinner lines
    else:
        margin = random.randint(2, 6)
        thickness = random.randint(2, 5)
    
    # Checkmark is made of two lines forming a "√" shape
    # First line: from left-middle to bottom-middle (down and right)
    # Second line: from bottom-middle to top-right (up and right)
    
    # Calculate checkmark points with some variation
    left_x = margin + random.randint(0, 2)
    left_y = img_size // 2 + random.randint(-3, 3)
    
    bottom_x = img_size // 2 + random.randint(-2, 2)
    bottom_y = img_size - margin - random.randint(0, 3)
    
    right_x = img_size - margin - random.randint(0, 2)
    right_y = margin + random.randint(0, 3)
    
    # Draw the checkmark with thickness
    for t in range(thickness):
        # First stroke: down and right
        draw.line(
            (left_x - t, left_y, bottom_x, bottom_y + t),
            fill=255,
            width=1,
        )
        # Second stroke: up and right
        draw.line(
            (bottom_x, bottom_y - t, right_x + t, right_y),
            fill=255,
            width=1,
        )
    
    # Apply rotation
    angle = random.uniform(-15, 15)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    return canvas

def normalize_and_post(img: Image.Image) -> Image.Image:
    # Apply autocontrast and noise
    img = ImageOps.autocontrast(img, cutoff=1)
    img = add_noise(img, amount=random.uniform(0.0, 0.01))
    return img

def save_samples(out_dir: Path, cls_name: str, count: int, img_size: int):
    for i in range(count):
        # Mix regular and small variants (30% small variants)
        small_variant = random.random() < 0.3
        
        if cls_name == "O":
            img = draw_O(img_size, small_variant)
        elif cls_name == "X":
            img = draw_X(img_size, small_variant)
        else:  # checkmark
            img = draw_checkmark(img_size, small_variant)
        img = normalize_and_post(img)
        (out_dir / cls_name).mkdir(parents=True, exist_ok=True)
        img.save(out_dir / cls_name / f"{i:06d}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="output dataset dir")
    parser.add_argument("--train", type=int, default=30000, help="number of training samples (default: 30000)")
    parser.add_argument("--val", type=int, default=6000, help="number of validation samples (default: 6000)")
    parser.add_argument("--img-size", type=int, default=28)
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dir(out / "train" / "O")
    ensure_dir(out / "train" / "X")
    ensure_dir(out / "train" / "checkmark")
    ensure_dir(out / "val" / "O")
    ensure_dir(out / "val" / "X")
    ensure_dir(out / "val" / "checkmark")

    print(f"Generating large-scale training set with {args.train} samples...")
    print("Including 30% small-sized shape variants for enhanced variety...")
    save_samples(out / "train", "O", args.train // 3, args.img_size)
    save_samples(out / "train", "X", args.train // 3, args.img_size)
    save_samples(out / "train", "checkmark", args.train // 3, args.img_size)

    print(f"Generating validation set with {args.val} samples...")
    save_samples(out / "val", "O", args.val // 3, args.img_size)
    save_samples(out / "val", "X", args.val // 3, args.img_size)
    save_samples(out / "val", "checkmark", args.val // 3, args.img_size)

    print("Dataset generation completed successfully!")
    print(f"Total training samples: {args.train}")
    print(f"Total validation samples: {args.val}")
    print(f"Dataset saved to: {args.out}")
    print("Enhanced with small-sized shape variants for improved recognition capability.")

if __name__ == "__main__":
    main()
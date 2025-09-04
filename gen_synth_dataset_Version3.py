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

def draw_O(img_size: int) -> Image.Image:
    canvas = Image.new("L", (img_size, img_size), color=0)  # black bg
    draw = ImageDraw.Draw(canvas)
    margin = random.randint(2, 6)
    x0 = margin + random.randint(0, 2)
    y0 = margin + random.randint(0, 2)
    x1 = img_size - margin - random.randint(0, 2)
    y1 = img_size - margin - random.randint(0, 2)
    thickness = random.randint(2, 5)
    # Draw outer ellipse by stroking multiple widths
    for t in range(thickness):
        draw.ellipse((x0 - t, y0 - t, x1 + t, y1 + t), outline=255)
    # Random rotation (draw on bigger canvas then rotate)
    angle = random.uniform(-20, 20)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    return canvas

def draw_X(img_size: int) -> Image.Image:
    canvas = Image.new("L", (img_size, img_size), color=0)  # black bg
    draw = ImageDraw.Draw(canvas)
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

def normalize_and_post(img: Image.Image) -> Image.Image:
    # Slight blur via resize trick and clamp
    img = img.filter(Image.Image.filter if hasattr(Image.Image, "filter") else lambda x: x)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = add_noise(img, amount=random.uniform(0.0, 0.01))
    return img

def save_samples(out_dir: Path, cls_name: str, count: int, img_size: int):
    for i in range(count):
        if cls_name == "O":
            img = draw_O(img_size)
        else:
            img = draw_X(img_size)
        img = normalize_and_post(img)
        (out_dir / cls_name).mkdir(parents=True, exist_ok=True)
        img.save(out_dir / cls_name / f"{i:06d}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="output dataset dir")
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--val", type=int, default=800)
    parser.add_argument("--img-size", type=int, default=28)
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dir(out / "train" / "O")
    ensure_dir(out / "train" / "X")
    ensure_dir(out / "val" / "O")
    ensure_dir(out / "val" / "X")

    print("Generating training set...")
    save_samples(out / "train", "O", args.train // 2, args.img_size)
    save_samples(out / "train", "X", args.train // 2, args.img_size)

    print("Generating validation set...")
    save_samples(out / "val", "O", args.val // 2, args.img_size)
    save_samples(out / "val", "X", args.val // 2, args.img_size)

    print("Done.")

if __name__ == "__main__":
    main()
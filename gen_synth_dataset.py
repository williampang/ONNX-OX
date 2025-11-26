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


def draw_V(img_size: int) -> Image.Image:
    canvas = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(canvas)
    thickness = random.randint(2, 4)
    left_x = random.randint(2, 6)
    left_y = random.randint(img_size // 4, img_size // 2)
    bottom_x = img_size // 2 + random.randint(-2, 3)
    bottom_y = img_size - random.randint(2, 5)
    right_x = img_size - random.randint(2, 6)
    right_y = random.randint(2, img_size // 3)
    draw.line((left_x, left_y, bottom_x, bottom_y), fill=255, width=thickness)
    draw.line((bottom_x, bottom_y, right_x, right_y), fill=255, width=thickness)
    angle = random.uniform(-10, 10)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    return canvas

def normalize_and_post(img: Image.Image) -> Image.Image:
    # Optional slight blur; original implementation attempted to call Image.filter incorrectly.
    try:
        from PIL import ImageFilter
        # Apply a very light random Gaussian blur to introduce variation.
        radius = random.uniform(0.0, 0.7)
        if radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    except Exception:
        pass
    img = ImageOps.autocontrast(img, cutoff=1)
    img = add_noise(img, amount=random.uniform(0.0, 0.01))
    return img

DRAW_FUNCS = {
    "O": draw_O,
    "X": draw_X,
    "V": draw_V,
}


def save_samples(out_dir: Path, cls_name: str, count: int, img_size: int):
    drawer = DRAW_FUNCS.get(cls_name)
    if drawer is None:
        raise ValueError(f"No drawing function defined for class '{cls_name}'")
    cls_dir = out_dir / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = drawer(img_size)
        img = normalize_and_post(img)
        img.save(cls_dir / f"{i:06d}.png")


def split_counts(total: int, num_classes: int):
    base = total // num_classes
    remainder = total % num_classes
    counts = [base] * num_classes
    for i in range(remainder):
        counts[i] += 1
    return counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="output dataset dir")
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--val", type=int, default=800)
    parser.add_argument("--img-size", type=int, default=28)
    parser.add_argument(
        "--classes",
        type=str,
        default="O,X,V",
        help="Comma-separated class labels to synthesize",
    )
    args = parser.parse_args()

    out = Path(args.out)
    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    if not classes:
        raise ValueError("At least one class label must be provided")

    for split in ("train", "val"):
        for cls in classes:
            ensure_dir(out / split / cls)

    def generate_split(split_name: str, total_count: int):
        print(f"Generating {split_name} set...")
        counts = split_counts(total_count, len(classes))
        for cls, cnt in zip(classes, counts):
            save_samples(out / split_name, cls, cnt, args.img_size)

    generate_split("train", args.train)
    generate_split("val", args.val)

    print("Done.")

if __name__ == "__main__":
    main()
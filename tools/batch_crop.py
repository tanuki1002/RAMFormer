"""Crop all images in a folder to the same region.

Usage:
    python tools/batch_crop.py --input <input_dir> --output <output_dir>
    python tools/batch_crop.py --input <input_dir> --output <output_dir> --box x,y,w,h

If --box is not given, a window opens on the first image so you can drag a
rectangle to select the crop region (press ENTER/SPACE to confirm, 'c' to
cancel, ESC to abort). The same region is then applied to every image found
in --input and saved under --output with the original filenames.
"""

import argparse
import os
import sys

import cv2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description="Crop all images in a folder to the same region.")
    parser.add_argument("--input", "-i", required=True, help="Folder containing input images.")
    parser.add_argument("--output", "-o", required=True, help="Folder to save cropped images.")
    parser.add_argument(
        "--box",
        type=str,
        default=None,
        help="Optional crop box as x,y,w,h. Skips interactive selection if given.",
    )
    return parser.parse_args()


def list_images(folder):
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTS)]
    return files


def select_box(image_path):
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"Failed to read {image_path} for ROI selection.")

    window = "Select crop region - ENTER/SPACE to confirm, C to cancel"
    x, y, w, h = cv2.selectROI(window, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)

    if w == 0 or h == 0:
        sys.exit("No region selected, aborting.")

    return x, y, w, h


def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Input folder not found: {args.input}")

    files = list_images(args.input)
    if not files:
        sys.exit(f"No images found in {args.input}")

    if args.box:
        try:
            x, y, w, h = (int(v) for v in args.box.split(","))
        except ValueError:
            sys.exit("--box must be formatted as x,y,w,h")
    else:
        first_image_path = os.path.join(args.input, files[0])
        x, y, w, h = select_box(first_image_path)
        print(f"Selected crop box: x={x}, y={y}, w={w}, h={h}")

    os.makedirs(args.output, exist_ok=True)

    saved, skipped = 0, 0
    for name in files:
        src_path = os.path.join(args.input, name)
        image = cv2.imread(src_path)
        if image is None:
            print(f"Skipping unreadable file: {name}")
            skipped += 1
            continue

        img_h, img_w = image.shape[:2]
        x2, y2 = min(x + w, img_w), min(y + h, img_h)
        if x >= img_w or y >= img_h or x2 <= x or y2 <= y:
            print(f"Skipping {name}: crop box out of bounds ({img_w}x{img_h})")
            skipped += 1
            continue

        cropped = image[y:y2, x:x2]
        cv2.imwrite(os.path.join(args.output, name), cropped)
        saved += 1

    print(f"Done. Saved {saved} images to {args.output} ({skipped} skipped).")


if __name__ == "__main__":
    main()

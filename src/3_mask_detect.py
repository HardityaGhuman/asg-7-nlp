"""Objective 3: detect faces in mask-wearing images; compare Haar vs YuNet.

Put images in data/mask/ (or capture via: python src/1_image_capture.py --mask).
Usage: python src/3_mask_detect.py
Saves side-by-side images to outputs/mask/ and prints a results table.
"""
import cv2
import numpy as np

from detectors import OUT, ROOT, RED, HaarDetector, YuNetDetector, draw

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    in_dir = ROOT / "data" / "mask"
    out_dir = OUT / "mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No images in {in_dir}")
    haar, yunet = HaarDetector(), YuNetDetector()

    totals = {"Haar": 0, "YuNet": 0}
    print(f"{'image':<40}{'Haar':>6}{'YuNet':>7}")
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h_boxes, y_boxes = haar.detect(img), yunet.detect(img)
        totals["Haar"] += len(h_boxes)
        totals["YuNet"] += len(y_boxes)
        side = np.hstack([draw(img, h_boxes, RED, "Haar"), draw(img, y_boxes, label="YuNet")])
        cv2.imwrite(str(out_dir / f"{p.stem}_compare.jpg"), side)
        print(f"{p.name:<40}{len(h_boxes):>6}{len(y_boxes):>7}")
    print(f"{'TOTAL faces detected':<40}{totals['Haar']:>6}{totals['YuNet']:>7}")
    print(f"Side-by-side outputs in {out_dir}")


if __name__ == "__main__":
    main()

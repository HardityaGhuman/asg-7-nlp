"""Objective 1: capture still images from webcam and detect faces.

Live preview window. SPACE = capture + detect + save, Q = quit.
Usage: python src/1_image_capture.py [--mask]   (--mask saves raw frames to data/mask/ too)
"""
import sys
import time

import cv2

from detectors import OUT, ROOT, YuNetDetector, HaarDetector, draw, open_webcam, RED


def main():
    mask_mode = "--mask" in sys.argv
    out_dir = OUT / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    haar, yunet = HaarDetector(), YuNetDetector()
    cap = open_webcam()
    print("SPACE = capture, Q = quit")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Webcam (SPACE capture, Q quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            ts = f"{'mask_' if mask_mode else ''}{time.strftime('%Y%m%d_%H%M%S')}_{n + 1}"
            y_boxes, h_boxes = yunet.detect(frame), haar.detect(frame)
            result = draw(frame, y_boxes, label="YuNet")
            cv2.imwrite(str(out_dir / f"capture_{ts}_yunet.jpg"), result)
            cv2.imwrite(str(out_dir / f"capture_{ts}_haar.jpg"), draw(frame, h_boxes, RED, "Haar"))
            if mask_mode:
                cv2.imwrite(str(ROOT / "data" / "mask" / f"webcam_{ts}.jpg"), frame)
            cv2.imshow("Detection", result)
            n += 1
            print(f"[{n}] saved capture_{ts}: YuNet={len(y_boxes)} Haar={len(h_boxes)} faces")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Shared face detectors: Haar cascade (classical baseline) and YuNet (CNN)."""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

GREEN, RED = (0, 200, 0), (0, 0, 255)


class HaarDetector:
    name = "Haar"

    def __init__(self):
        self.model = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, img):
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        boxes = self.model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [(int(x), int(y), int(w), int(h), None) for x, y, w, h in boxes]


class YuNetDetector:
    name = "YuNet"

    def __init__(self, score_threshold=0.7):
        self.model = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold)

    def detect(self, img):
        h, w = img.shape[:2]
        self.model.setInputSize((w, h))
        _, faces = self.model.detect(img)
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[14])) for f in faces]


def draw(img, boxes, color=GREEN, label=""):
    out = img.copy()
    for x, y, w, h, score in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if score is not None:
            cv2.putText(out, f"{score:.2f}", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(out, f"{label} faces: {len(boxes)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def open_webcam(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (check camera permission for your terminal/IDE).")
    return cap

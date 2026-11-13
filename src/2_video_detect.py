"""Objective 2: record a short live webcam sequence and detect faces per frame.

Usage: python src/2_video_detect.py [seconds=10]
Saves annotated video to outputs/video/ and prints per-frame stats + FPS.
"""
import sys
import time

import cv2

from detectors import OUT, YuNetDetector, draw, open_webcam


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    out_dir = OUT / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = YuNetDetector()
    cap = open_webcam()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Webcam returned no frame.")
    h, w = frame.shape[:2]
    path = out_dir / f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

    frames = with_face = 0
    det_time = 0.0
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        boxes = detector.detect(frame)
        det_time += time.perf_counter() - t0
        frames += 1
        with_face += bool(boxes)
        annotated = draw(frame, boxes, label="YuNet")
        writer.write(annotated)
        cv2.imshow("Live video detection (Q quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"Saved {path}")
    print(f"Frames: {frames} in {elapsed:.1f}s ({frames / elapsed:.1f} FPS overall)")
    print(f"Detection: {1000 * det_time / max(frames, 1):.1f} ms/frame")
    print(f"Frames with >=1 face: {with_face}/{frames} ({100 * with_face / max(frames, 1):.0f}%)")


if __name__ == "__main__":
    main()

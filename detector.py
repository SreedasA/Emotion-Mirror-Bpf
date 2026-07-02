import cv2
import mediapipe as mp

_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)


def detect_faces(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = _detector.process(rgb)
    if not out.detections:
        return []

    h, w = frame.shape[:2]
    boxes = []
    for det in out.detections:
        b = det.location_data.relative_bounding_box
        x_raw, y_raw = b.xmin * w, b.ymin * h
        w_raw, h_raw = b.width * w, b.height * h

        x = max(int(x_raw), 0)
        y = max(int(y_raw), 0)
        # shrink the box if the face clips the frame edge rather than just
        # clamping the origin -- otherwise the crop comes out wrong
        bw = max(min(int(w_raw - (x - x_raw)), w - x), 0)
        bh = max(min(int(h_raw - (y - y_raw)), h - y), 0)

        if bw > 0 and bh > 0:
            boxes.append((x, y, bw, bh))

    return boxes

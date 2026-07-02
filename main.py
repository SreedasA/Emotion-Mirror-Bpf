import cv2
from detector import detect_faces
from classifier import classify, Smoother

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLORS = {
    "happy": (50, 220, 100),
    "sad": (200, 100, 50),
    "angry": (50, 50, 220),
    "surprised": (50, 200, 200),
    "neutral": (160, 160, 160),
    "uncertain": (100, 100, 100),
}

# match faces across frames by center distance so each face keeps its
# own smoothing history -- not a proper tracker but good enough here
MATCH_DIST = 80


def _match(center, prev_faces):
    best, best_d = None, MATCH_DIST ** 2
    cx, cy = center
    for f in prev_faces:
        dx = f["cx"] - cx
        dy = f["cy"] - cy
        d = dx*dx + dy*dy
        if d < best_d:
            best, best_d = f, d
    return best


def draw(frame, detections):
    if not detections:
        msg = "Looking for a face..."
        tw = cv2.getTextSize(msg, FONT, 0.7, 2)[0][0]
        x = (frame.shape[1] - tw) // 2
        y = frame.shape[0] // 2
        cv2.putText(frame, msg, (x, y), FONT, 0.7, (160, 160, 160), 2)
        return

    for d in detections:
        x, y, w, h = d["box"]
        label = d["label"]
        conf  = d["conf"]
        color = COLORS.get(label, (180, 180, 180))

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        text = f"{label}  {int(conf*100)}%"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        ty = max(y - 8, th + 4)
        cv2.rectangle(frame, (x, ty-th-6), (x+tw+8, ty+2), color, -1)
        cv2.putText(frame, text, (x+4, ty-2), FONT, 0.6, (0, 0, 0), 1)

        bar_y = y + h + 5
        cv2.rectangle(frame, (x, bar_y), (x+w, bar_y+5), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, bar_y), (x + int(w*conf), bar_y+5), color, -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("couldn't open webcam")
        return

    smoother = Smoother(window=6)
    prev_faces = []

    print("Emotion Mirror  |  press q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes = detect_faces(frame)
        curr_faces = []

        for box in boxes:
            x, y, w, h = box
            cx, cy = x + w/2, y + h/2

            matched = _match((cx, cy), prev_faces)
            face_id = matched["id"] if matched else id((cx, cy))

            # add padding so the model sees forehead and chin, not just the outline
            fh, fw = frame.shape[:2]
            pad = max(10, int(min(w, h) * 0.18))
            crop = frame[max(0,y-pad):min(fh,y+h+pad), max(0,x-pad):min(fw,x+w+pad)]

            label, conf = classify(crop, face_id, smoother)
            curr_faces.append({"id": face_id, "cx": cx, "cy": cy,
                                "box": box, "label": label, "conf": conf})

        # clear smoother buffers for faces that left the frame
        curr_ids = {f["id"] for f in curr_faces}
        for f in prev_faces:
            if f["id"] not in curr_ids:
                smoother.clear(f["id"])

        prev_faces = curr_faces

        draw(frame, curr_faces)
        cv2.imshow("Emotion Mirror", frame)

        # 30ms is more than enough -- the model takes longer than this anyway
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

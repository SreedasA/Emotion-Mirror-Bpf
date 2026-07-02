import cv2
import numpy as np
from collections import deque
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# enet_b0_8_best_afew is trained on acted facial expressions in the wild
# which matches live webcam use way better than the AffectNet models
_model = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")

# the model outputs 8 classes, we only show 5
# contempt looks almost identical to anger on camera so it folds in
# fear maps to surprised -- raised brows + wide eyes is the same expression
# disgust doesn't cleanly belong to any one bucket (confused with anger, fear,
# sad, and neutral about equally in the literature), so it falls back to
# neutral rather than artificially inflating "angry"
LABEL_MAP = {
    "Anger": "angry",
    "Contempt": "angry",
    "Disgust": "neutral",
    "Fear": "surprised",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprised",
}

LABELS = ["angry", "happy", "neutral", "sad", "surprised"]

# 0.30 felt right after testing -- below this the model is basically
# split between two emotions and the label just flickers
THRESHOLD = 0.30


def _merge_probs(raw_scores):
    merged = {l: 0.0 for l in LABELS}
    for idx, score in enumerate(raw_scores):
        target = LABEL_MAP[_model.idx_to_class[idx]]
        merged[target] += float(score)
    total = sum(merged.values())
    return np.array([merged[l] / total for l in LABELS], dtype=np.float32)


class Smoother:
    # averages probability vectors over the last N frames so the label doesn't
    # jump around on every frame -- 6 frames at 10fps is about 0.6 seconds
    def __init__(self, window=6):
        self._bufs = {}
        self.window = window

    def push(self, face_id, probs5):
        if face_id not in self._bufs:
            self._bufs[face_id] = deque(maxlen=self.window)
        self._bufs[face_id].append(probs5)

    def clear(self, face_id):
        self._bufs.pop(face_id, None)

    def result(self, face_id):
        if face_id not in self._bufs or not self._bufs[face_id]:
            return "uncertain", 0.0
        avg = np.mean(self._bufs[face_id], axis=0)
        idx = int(np.argmax(avg))
        conf = float(avg[idx])
        return (LABELS[idx] if conf >= THRESHOLD else "uncertain"), conf


def classify(face_bgr, face_id, smoother):
    # model expects RGB, frame is BGR
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    _, raw = _model.predict_emotions(rgb, logits=False)
    smoother.push(face_id, _merge_probs(raw))
    return smoother.result(face_id)

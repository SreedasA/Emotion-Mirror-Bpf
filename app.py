import asyncio
import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from detector import detect_faces
from classifier import classify, Smoother

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


# single smoother for the session -- fine for a local single-user demo
_smoother = Smoother(window=6)
_prev_faces = []

# match faces across frames by center distance so each face keeps its own
# smoothing history instead of one keyed by frame-local list position
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


def _run(frame):
    global _prev_faces
    results = []
    boxes = detect_faces(frame)
    fh, fw = frame.shape[:2]
    curr_faces = []

    for (x, y, w, h) in boxes:
        cx, cy = x + w/2, y + h/2
        pad = max(10, int(min(w, h) * 0.18))
        crop = frame[max(0,y-pad):min(fh,y+h+pad), max(0,x-pad):min(fw,x+w+pad)]
        if crop.size == 0:
            continue

        matched = _match((cx, cy), _prev_faces)
        face_id = matched["id"] if matched else id((cx, cy))

        label, conf = classify(crop, face_id, smoother=_smoother)
        curr_faces.append({"id": face_id, "cx": cx, "cy": cy})
        results.append({"bbox": [x, y, w, h], "emotion": label, "confidence": round(conf, 3)})

    curr_ids = {f["id"] for f in curr_faces}
    for f in _prev_faces:
        if f["id"] not in curr_ids:
            _smoother.clear(f["id"])

    _prev_faces = curr_faces
    return results


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    log.info("connected")
    try:
        while True:
            data = await ws.receive_text()
            if "," in data:
                data = data.split(",", 1)[1]
            try:
                buf = np.frombuffer(base64.b64decode(data), np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                await ws.send_text("[]")
                continue
            if frame is None:
                await ws.send_text("[]")
                continue
            # learned the hard way that running inference on the event loop
            # makes the websocket unresponsive -- thread it
            result = await asyncio.to_thread(_run, frame)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        log.info("disconnected")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

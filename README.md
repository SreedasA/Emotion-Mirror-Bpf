# Emotion Mirror

Webcam app that reads your face in real time and labels what emotion you're making. Runs in the browser, nothing to install beyond Python.

## Setup

Python 3.11.

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000, allow camera access, and you're done. The first run downloads the emotion model (~25 MB) and caches it. After that it starts instantly.

That's the main submission.

There's also a standalone OpenCV version that skips the browser and opens a desktop window:

```bash
python main.py  # press q to quit
```

Uses the same detector and classifier, just no server or WebSocket. Good for a quick check if you don't want to open a browser.

## Why these tools

I picked MediaPipe for face detection because it's fast enough on CPU that it doesn't add noticeable lag. Haar cascades are faster but miss faces at angles and in lower light. Heavier detectors like RetinaFace or MTCNN felt like overkill for a single webcam feed.

For the emotion model I used `enet_b0_8_best_afew` from hsemotion-onnx. The main reason is no TensorFlow dependency — on Windows, `pip install tensorflow` installs a redirect stub that crashes on import. The "afew" part is the dataset it was trained on: acted facial expressions in the wild. That works better for a live webcam demo than models trained on posed photos in a lab.

## How it works

The browser captures your webcam and sends JPEG frames over a WebSocket at around 10 fps. The server decodes each frame, finds faces with MediaPipe, crops them, and runs the emotion classifier. Results go back to the browser as JSON and get drawn as overlays on the video.

Emotion mapping — the model outputs 8 classes but I only show 5:

| model class | shown as  | reason |
|-------------|-----------|--------|
| Anger       | angry     | direct |
| Contempt    | angry     | same furrowed-brow expression on camera |
| Disgust     | neutral   | kept getting confused with angry, moved it here instead |
| Fear        | surprised | raised brows + wide eyes looks like surprise |
| Happiness   | happy     | direct |
| Neutral     | neutral   | direct |
| Sadness     | sad       | direct |
| Surprise    | surprised | direct |

If smoothed confidence drops below 30%, it shows "uncertain" instead of flickering between two close guesses.

## Project structure

```
app.py              web server (FastAPI + WebSocket)
main.py             standalone OpenCV demo, no browser needed
detector.py         MediaPipe face detection
classifier.py       emotion model + smoothing
static/index.html   single-page frontend
requirements.txt
```

## What I found challenging

The biggest thing was making the labels stable. The raw model output flips around on every frame even when your expression hasn't changed: lighting, slight head movement, JPEG compression all affect it. I ended up averaging the probability vectors over the last 6 frames rather than just taking the top prediction each frame. That made a huge difference. Voting on labels is worse because if angry and neutral each win 3 frames you get a permanent tie that flips; averaging the underlying numbers gives a clear winner.

The other thing that caught me out was the coordinate flip. The browser mirrors the video with CSS so it looks natural, but the server gets the raw unflipped frame. Bounding boxes come back in raw coordinates, so x has to be flipped before drawing on the canvas. I had the boxes on the wrong side for a while before I figured that out.

One annoying dependency issue: mediapipe removed the `mp.solutions` API in version 0.10.15. I'm pinned to 0.10.14 in requirements.txt for that reason.

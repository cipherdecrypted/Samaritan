# Gotham

A local, GPU-accelerated computer-vision workstation with a Samaritan-style interface and an AI observer you talk
to in plain language. Faces are detected, tracked and remembered across sessions by **GothamFaces**, Gotham's
recognition engine. Everything stays on this machine, encrypted.

## Run

```
Gotham.bat                                  fullscreen
Gotham.bat --camera 0                       open camera 0 at start
Gotham.bat --file clip.mp4 --skip-boot      a video file, no boot sequence
Gotham.bat --stream rtsp://host/stream      a network stream
Gotham.bat --images C:\photos               photos for the image lab
Gotham.bat --windowed                       borderless window instead of fullscreen
```

Setup (Windows, Python 3.12, a DirectX 12 GPU):

```
setup.bat
```

It creates the environment, installs the packages, downloads every model from its official source (each checked
against a fixed SHA-256) and builds the GothamFaces model files. The build step uses a temporary environment that is
removed afterwards. About 1.2 GB of downloads; run it again at any time to repair a missing or damaged file.

## What it does

| Area | |
|---|---|
| Sources | Webcams and USB cameras (discovered with their real names), RTSP/HTTP streams with auto-reconnect, video files, photos and folders. Any number of feeds at once. |
| Detection | GPU face detection with a round-robin *deep scan* for distant faces; *small-face mode* scans every tile every frame. |
| Landmarks | 478-point face mesh with irises and expressions; head pose, gaze, eye openness, proportions, image quality. |
| Tracking | Motion prediction with appearance tie-breaks where people cross, coasting through occlusion, re-acquisition. |
| GothamFaces | Two-network fused face descriptor, strongest on small and low-quality faces. Each face sample is weighted by how recognisable it is; a track is matched as a whole. |
| Face memory | Persistent profiles (P-0001 …) with a photo for every learned head angle, visit history, cross-camera transit, possible sightings, look-alikes, duplicate review, watchlist alerts. Labels and notes come only from the operator. |
| Profile pages | Identity and stats, face design (the stored mesh as a wireframe, proportions, angle coverage), learned views with how well each fits, highest-confidence sightings, possible sightings to confirm or reject, time-of-day and 14-day activity, full timeline, often-seen-with, look-alikes, notes. |
| Display | Video at the camera's full frame rate with overlays following each subject; reticles, face web, gaze, pose, trails, dossiers, scan sweep; Samaritan and Machine styles. |
| Observer | A Samaritan persona over OpenRouter or Anthropic that reads structured observations (never images) and can query memory, arm watches, start tracking, focus subjects and switch display layers. |

## Commands

Type in the bar under the video (`/` focuses it). Plain language goes to the observer, for example
`how many people are on feed 1?` or `alert me when P-0042 is seen`.

Local commands: `:web` `:mesh` `:contours` `:eyes` `:gaze` `:pose` `:trails` `:labels` `:grid` `:telemetry` (append
`on`/`off`), `:samaritan`, `:machine`, `:track on|off`, `:reset`, `:snap`, `:key <api key>`, `:model <openrouter model>`,
`:recogniser full|lite` (lite is faster in dense crowds; applies at next start, memory converts automatically).

## Face memory

Stored in `%LOCALAPPDATA%\Gotham\memory`, encrypted with Windows DPAPI (readable only by this Windows user).

- Sightings are visits: fragments of one person on one camera less than 60 s apart are one visit.
- Stored views that clearly show someone else are removed when memory opens; each profile page shows how well
  every view fits and lets you remove one.
- MERGE N CERTAIN folds duplicate profiles that match as wholes into the oldest one; other pairs are listed for review.
- Changing the recognition engine keeps every profile: descriptors are re-made in the background from stored photos,
  after a backup `memory.before-<model>.db`.

## Performance

i7-12700KF + RX 7600 XT, 480 stored profiles, 1280×720 crowd video:

| Scene | Analysis | Video |
|---|---|---|
| ~13 faces | 15–19 fps | camera rate |
| ~3 faces | every frame of a 30 fps source | camera rate |

## Development

```
.venv\Scripts\python -m unittest discover -s tests -t .
```

`tools/compare_recognisers.py` benchmarks recognition on LFW pairs; `tools/make_batch_arcface.py` and
`tools/export_adaface.py` rebuild the GothamFaces model files (`pip install -r requirements-tools.txt`).

Licences: `LICENSES/`.

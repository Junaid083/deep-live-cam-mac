# Deep-Live-Cam (macOS edition)

Real-time AI face swap from a single photo, live in Google Meet, Zoom, Teams or any app that lets you pick a camera. Tuned for Apple Silicon Macs.

A trimmed fork of [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) with:

- **Direct meeting output:** Live mode streams straight to OBS Virtual Camera (no window capture).
- **One-command setup:** `./setup.sh` installs everything and the `deeplive` command.
- **Steadier, more precise tracking:** your face is detected on every frame and smoothed, so the swap doesn't shimmer.
- **Reliable face changes:** picking a new photo, or 🔄 random faces, switch live; close-up portraits are detected too.
- **Health line in the terminal** every 2 s: face found, frames swapped, speed.
- Optional, experimental: `--head hair|full` adds the photo's hair.

## Setup

Needs an Apple Silicon Mac with [Homebrew](https://brew.sh).

```bash
./setup.sh          # add --all for the GFPGAN enhancer and hair model
```

Re-run it any time; it skips what's done and resumes downloads. Models go to `models/` and `~/.insightface/` (~650 MB).

For meetings, install OBS once (`brew install --cask obs`), open it, click **Start Virtual Camera**, allow it in **System Settings → General → Login Items & Extensions → Camera Extensions**, then quit OBS.

## Use

```bash
deeplive                      # open the app
deeplive ~/Desktop/face.jpg   # open with that face already picked
```

1. **Select a face**: a clear, front-facing, evenly lit photo.
2. Pick your camera and click **Live**. The first start takes 10–30 s.
3. In the meeting app, choose **OBS Virtual Camera**.

You don't need **Select a target**, **Start** or **Preview** for live use. They make edited photo/video files.

## Best settings for calls

| Setting | Value |
| --- | --- |
| Sharpness | ~1.0 (crisper face, free) |
| Poisson Blend | On |
| Mouth Mask | 0–20 |
| Face Enhancer | None (GPEN-256 is sharper but ~6 fps) |
| Det size | 640, or 320 if it lags |

A face with a similar hairstyle and beard to yours looks the most convincing, and so does light in front of you. If both of you wear glasses the eyes can glitch, so use a photo without glasses.

## Troubleshooting

- **The terminal says "no face seen":** face the camera, add light, or set Det size to 320.
- **"No face found in <photo>":** pick a sharper, front-facing photo.
- **Not listed in Meet:** finish the OBS step above, then restart the browser.
- `E5RT` / `Screenshot + Copy` messages are harmless macOS notices.

## Responsible use

Use your own face, or faces of people who agreed. Don't impersonate real people or use it to deceive anyone. Label deepfake content as such.

## Licence

AGPL-3.0, see [LICENSE](LICENSE). Based on Deep-Live-Cam by hacksider and contributors. The face-swap model (insightface `inswapper`) is for **non-commercial use only**.

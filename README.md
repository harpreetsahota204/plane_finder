# Plane Finder


<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

**The open-source tool for building high-quality datasets and computer vision
models**

---

<!-- prettier-ignore -->
<a href="https://voxel51.com/fiftyone?utm_source=harpreet-gh">Website</a> •
<a href="https://docs.voxel51.com?utm_source=harpreet-gh">Docs</a> •
<a href="https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb?utm_source=harpreet-gh">Try it Now</a> •
<a href="https://docs.voxel51.com/getting_started_guides/index.html?utm_source=harpreet-gh">Getting Started Guides</a> •
<a href="https://docs.voxel51.com/tutorials/index.html?utm_source=harpreet-gh">Tutorials</a> •
<a href="https://voxel51.com/blog/?utm_source=harpreet-gh">Blog</a> •
<a href="https://discord.gg/fiftyone-community?utm_source=harpreet-gh">Community</a>

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</p>
</div>

![image/png](plane_finder.gif)

A [FiftyOne](https://docs.voxel51.com) plugin that flags **raw carbon-fiber RC aircraft** in large aerial survey datasets — dark, geometrically regular, elongated objects sitting on noisy natural terrain.

It is a classical computer-vision detector (OpenCV, no training required) wrapped as two FiftyOne operators so you can run it, tune it, and review the results entirely inside the FiftyOne App.

---

## Why this exists

On uniform terrain (e.g. desert scrub) a glint- or color-based detector fails — the background is near-uniform mid-gray and the dark features are mostly vegetation. But an engineered carbon surface is still discriminable if you score **shape and surface character** instead of tone alone. This plugin scores each dark blob on four signals that separate an airframe from a bush:

| Signal | Airframe | Vegetation |
|---|---|---|
| **Aspect ratio** | elongated, 3:1+ (wingspan ≫ width) | clustered, ~1.5:1 |
| **Rectangle fill** | solid fuselage fills its box | leafy, full of gaps |
| **Interior uniformity** | smooth surface (low std dev) | noisy, lighter gaps between leaves |
| **Solidity** | convex, solid | spiky / branchy |

Each candidate gets a composite **score in `[0, 1]`** (higher = more aircraft-like), stored as the detection's `confidence`.

---

## How it works

For every image the detector:

1. **Thresholds** dark regions (`dark_threshold`).
2. **Morphologically closes** the mask to bridge small gaps (`morph_kernel`).
3. **Finds contours** and gates them by area and bounding-box diagonal.
4. **Measures** aspect ratio, rectangle fill, convex-hull solidity, and interior texture std.
5. **Hard-rejects** anything too round, too hollow, or too noisy.
6. **Scores** survivors with a weighted composite and writes them as `fo.Detections`.

### What it writes to your dataset

- **`<label_field>`** (default `plane_candidates`) — an `fo.Detections` field. Each `fo.Detection` carries `confidence` (the score) plus per-detection attributes: `rect_aspect`, `rect_fill`, `interior_std`, `interior_mean`, `solidity`, `area`, `bbox_diag`, `rect_angle`.
- **`max_<label_field>_score`** (e.g. `max_plane_candidates_score`) — a float per sample. **Sort by this descending** to put the most aircraft-like frames first.

Bounding boxes are stored in FiftyOne's normalized `[x, y, w, h]` (0–1) format, so they overlay correctly at any resolution.

---

## Operators

### 1. `find_plane` — the detector

Scans your samples and writes candidate detections + the ranking field. All thresholds are exposed as form inputs (each with an in-App description), so you never have to edit code to tune.

Key inputs: `dark_threshold`, `min_aspect_ratio`, `max_interior_std`, `min_rect_fill`, `min_diag_px`/`max_diag_px`, `min_area_px`, `morph_kernel`, plus output controls `min_score` and `max_per_image` to keep the field uncluttered.

### 2. `preview_dark_mask` — calibration helper

Before a full run, preview what `dark_threshold` actually captures. It writes the thresholded (optionally morph-closed) mask as an `fo.Heatmap` on a small sample of images. Toggle the heatmap overlay in the App, adjust the threshold, and re-run until the mask cleanly isolates dark objects. This is the App-native replacement for dumping debug mask JPEGs.

---

## Installation

Open the terminal and run:

```bash
fiftyone plugins download https://github.com/harpreetsahota204/plane_finder
```

Requirements: `opencv-python`, `numpy` (see `requirements.txt`). FiftyOne `>= 1.0`.

---

## Usage

### From the App

Open the operator browser (`` ` `` backtick), search **"Find Plane"** or **"Preview Dark Mask"**, fill in the form (every field has a description), and run. Then sort the grid by `max_plane_candidates_score` (descending) to triage from the top.

### From Python

```python
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.utils.huggingface import load_from_hub

# Other available arguments include 'max_samples', 'persistent', etc.
dataset = load_from_hub(
    "harpreetsahota/ariel_scans",
    persistent=True,
    overwrite=True
    )

preview = foo.get_operator("@harpreetsahota/plane_finder/preview_dark_mask")

find_plane = foo.get_operator("@harpreetsahota/plane_finder/find_plane")

# 1) Calibrate the threshold on a handful of images (view the dark_mask heatmap in the App)
await preview(dataset, num_samples=10, dark_threshold=105)

# 2) Full run — delegated, keeping only strong candidates
await find_plane(dataset, delegate=True, min_score=0.55, max_per_image=5)

# 3) Review, highest score first
session = fo.launch_app(dataset.sort_by("max_plane_candidates_score", reverse=True))
```

Filter on the per-detection attributes too:

```python
from fiftyone import ViewField as F
strong = dataset.filter_labels("plane_candidates", F("confidence") > 0.6)
```

---

## Optional delegation

Both operators have a **Delegate execution** checkbox (default **off**):

- **Off** — runs immediately with a live progress bar. Best for small views and calibration.

- **On** — queues a background job; process it with `fiftyone delegated launch` in a terminal. Recommended for large datasets (thousands of full-res frames).

---

## Tuning guide

Start here, in order:

1. **`dark_threshold`** — the single most important knob. Calibrate it with `preview_dark_mask`. If your survey altitude differs from the sample this was tuned on, this is the first thing to change.

2. **`min_diag_px` / `max_diag_px`** — set these to the expected on-screen size of the airframe in pixels. On high-resolution frames the defaults (28–400) are likely too small; widen the gate to match your imagery.

3. **`min_aspect_ratio`** — raise toward 3.0+ for a clearly elongated wing; lower toward 2.0 if the airframe can appear foreshortened or partially obscured.

4. **`max_interior_std`** — lower (toward ~12) to demand a smoother surface and reject noisy vegetation; raise if the target's surface is textured.

5. **`min_score` / `max_per_image`** — once tuned, use these to write only the best candidates per frame so the App stays clean.

---

## Notes & limitations

- **Classical CV, not ML** — no model, no training, fast and fully interpretable, but tuning-sensitive. It is a triage tool to shrink thousands of frames down to a reviewable shortlist, not an autonomous classifier.

- **Mixed-resolution datasets** — if your data contains both high-res RGB and small thermal/IR frames, the pixel-based size gates behave differently across them. Consider tagging them into separate views and running with resolution-appropriate settings.

- **Unreadable files** — formats OpenCV can't decode (e.g. HEIC) are skipped and recorded with an empty detections list and a score of 0.

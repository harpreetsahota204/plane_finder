"""
@harpreetsahota/plane_finder

A FiftyOne operator that flags dark, geometrically regular, elongated blobs as
candidate raw-carbon-fiber aircraft in aerial survey imagery.

Ported from the standalone ``find_plane.py`` script. Instead of emitting a CSV +
contact sheet, this writes one ``fo.Detections`` field per sample (score stored in
``confidence``) plus a sample-level ``max_<field>_score`` for ranking in the App.
"""

import cv2
import numpy as np

import fiftyone as fo
import fiftyone.core.view as fov
import fiftyone.operators as foo
import fiftyone.operators.types as types

# ─────────────────────────────────────────────
# TUNING DEFAULTS — every one of these is overridable from the operator form
# ─────────────────────────────────────────────
DEFAULTS = dict(
    dark_threshold=105,     # pixels darker than this are candidates (0-255)
    min_diag_px=28,         # min bounding-box diagonal
    max_diag_px=400,        # max bounding-box diagonal
    min_area_px=150,        # min blob area (filters sensor noise)
    min_aspect_ratio=2.0,   # wingspan >> width; veg is ~1.5:1
    max_interior_std=20.0,  # carbon is smooth (<13); veg is noisy (~18-20)
    min_rect_fill=0.40,     # solid fuselage fills its rectangle
    morph_kernel=5,         # close small gaps before contouring
)

# Scoring weights — higher composite score = more aircraft-like
W_ASPECT = 3.0
W_RECT_FILL = 2.0
W_UNIFORMITY = 2.5
W_SOLIDITY = 1.5
W_SIZE = 1.0
_TOTAL_W = W_ASPECT + W_RECT_FILL + W_UNIFORMITY + W_SOLIDITY + W_SIZE


def score_candidate(c, p):
    """Composite anomaly score in [0, 1]. Higher = more aircraft-like."""
    asp_score = min(c["rect_aspect"] / 5.0, 1.0)
    fill_score = min(c["rect_fill"] / 0.8, 1.0)
    max_std = p["max_interior_std"]
    uniformity_score = max(0.0, max_std - c["interior_std"]) / max_std
    solidity_score = c["solidity"]

    diag = c["bbox_diag"]
    target_diag = (p["min_diag_px"] + p["max_diag_px"]) / 2.0
    size_score = max(0.0, 1.0 - abs(diag - target_diag) / target_diag)

    total = (
        W_ASPECT * asp_score
        + W_RECT_FILL * fill_score
        + W_UNIFORMITY * uniformity_score
        + W_SOLIDITY * solidity_score
        + W_SIZE * size_score
    )
    return total / _TOTAL_W


def detect_candidates(gray, p):
    """Run the blob detector on a grayscale image.

    Returns a list of candidate dicts with bbox in ABSOLUTE pixel coords; the
    caller is responsible for normalizing to [0, 1] for FiftyOne.
    """
    _, dark_mask = cv2.threshold(gray, p["dark_threshold"], 255, cv2.THRESH_BINARY_INV)

    k = max(1, int(p["morph_kernel"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dark_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        dark_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < p["min_area_px"]:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        bbox_diag = float(np.sqrt(w**2 + h**2))
        if bbox_diag < p["min_diag_px"] or bbox_diag > p["max_diag_px"]:
            continue

        rect = cv2.minAreaRect(cnt)
        rect_w, rect_h = rect[1]
        if rect_w < 1 or rect_h < 1:
            continue
        rect_area = rect_w * rect_h
        rect_fill = area / (rect_area + 1e-5)
        rect_aspect = max(rect_w, rect_h) / (min(rect_w, rect_h) + 1e-5)

        if rect_aspect < p["min_aspect_ratio"]:
            continue
        if rect_fill < p["min_rect_fill"]:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-5)

        blob_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        interior_pixels = gray[blob_mask > 0]
        if len(interior_pixels) == 0:
            continue
        interior_std = float(interior_pixels.std())
        interior_mean = float(interior_pixels.mean())

        if interior_std > p["max_interior_std"]:
            continue

        c = dict(
            x=int(x), y=int(y), w=int(w), h=int(h),
            cx=int(x + w // 2), cy=int(y + h // 2),
            area=float(area),
            bbox_diag=bbox_diag,
            rect_aspect=float(rect_aspect),
            rect_fill=float(rect_fill),
            solidity=float(solidity),
            interior_std=interior_std,
            interior_mean=interior_mean,
            rect_angle=float(rect[2]),
        )
        c["score"] = float(score_candidate(c, p))
        candidates.append(c)

    return candidates


def _filter_and_cap(cands, min_score=0.0, max_per_image=0):
    """Drop low-scoring candidates and optionally keep only the top-N per image."""
    if min_score > 0.0:
        cands = [c for c in cands if c["score"] >= min_score]
    cands = sorted(cands, key=lambda c: c["score"], reverse=True)
    if max_per_image and max_per_image > 0:
        cands = cands[:max_per_image]
    return cands


def _candidates_to_detections(cands, width, height):
    """Convert absolute-pixel candidate dicts to a fo.Detections object."""
    dets = []
    for c in cands:
        dets.append(
            fo.Detection(
                label="plane_candidate",
                bounding_box=[
                    c["x"] / width,
                    c["y"] / height,
                    c["w"] / width,
                    c["h"] / height,
                ],
                confidence=c["score"],
                rect_aspect=c["rect_aspect"],
                rect_fill=c["rect_fill"],
                interior_std=c["interior_std"],
                interior_mean=c["interior_mean"],
                solidity=c["solidity"],
                area=c["area"],
                bbox_diag=c["bbox_diag"],
                rect_angle=c["rect_angle"],
            )
        )
    return fo.Detections(detections=dets)


def _execute(uri, sample_collection, params):
    """Shared SDK entry point: run an operator against a dataset or a view."""
    if isinstance(sample_collection, fov.DatasetView):
        ctx = dict(view=sample_collection)
    else:
        ctx = dict(dataset=sample_collection)
    return foo.execute_operator(uri, ctx, params=params)


class FindPlane(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="find_plane",
            label="Find Plane (carbon-fiber blob detector)",
            description="Flag dark, elongated, smooth blobs as aircraft candidates",
            dynamic=True,
            icon="/assets/icon.svg",
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # ── What to run on ──
        target_choices = types.RadioGroup()
        target_choices.add_choice("DATASET", label="Entire dataset")
        if ctx.view is not None and ctx.view != ctx.dataset.view():
            target_choices.add_choice("CURRENT_VIEW", label="Current view")
        if ctx.selected:
            target_choices.add_choice("SELECTED_SAMPLES", label="Selected samples")
        default_target = "CURRENT_VIEW" if ctx.view is not None else "DATASET"
        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            view=target_choices,
            label="Run on",
            description="Which samples to scan: the whole dataset, the current App view, or just your selected samples.",
        )

        inputs.str(
            "label_field",
            default="plane_candidates",
            required=True,
            label="Output detections field",
            description="Detections are written here; ranking goes to max_<field>_score",
        )

        # ── Detection knobs ──
        inputs.int(
            "dark_threshold",
            default=DEFAULTS["dark_threshold"],
            label="Dark threshold (0-255)",
            description="Pixels darker than this are candidates. First knob to tune.",
        )
        inputs.float(
            "min_aspect_ratio",
            default=DEFAULTS["min_aspect_ratio"],
            label="Min aspect ratio",
            description="Wingspan >> width. Vegetation is ~1.5:1; a wing is 3:1+.",
        )
        inputs.float(
            "max_interior_std",
            default=DEFAULTS["max_interior_std"],
            label="Max interior std",
            description="Smooth carbon <13; noisy vegetation ~18-20. Hard reject above this.",
        )
        inputs.float(
            "min_rect_fill",
            default=DEFAULTS["min_rect_fill"],
            label="Min rectangle fill",
            description=(
                "Fraction of its rotated bounding box the blob fills (0-1). A solid "
                "fuselage fills its box; a leafy bush has gaps. Higher = stricter."
            ),
        )
        inputs.int(
            "min_diag_px",
            default=DEFAULTS["min_diag_px"],
            label="Min bbox diagonal (px)",
            description=(
                "Smallest blob to consider, measured as bounding-box diagonal in pixels. "
                "Raise it on high-res frames to ignore tiny specks."
            ),
        )
        inputs.int(
            "max_diag_px",
            default=DEFAULTS["max_diag_px"],
            label="Max bbox diagonal (px)",
            description=(
                "Largest blob to consider (bounding-box diagonal in pixels). Caps out "
                "huge shadows/terrain features. Scale this with your survey altitude."
            ),
        )
        inputs.int(
            "min_area_px",
            default=DEFAULTS["min_area_px"],
            label="Min blob area (px)",
            description="Minimum filled area in pixels. Filters out sensor noise and dust specks.",
        )
        inputs.int(
            "morph_kernel",
            default=DEFAULTS["morph_kernel"],
            label="Morph close kernel",
            description=(
                "Size (px) of the morphological-close kernel applied before contouring. "
                "Larger values bridge gaps and merge broken-up pieces into one blob."
            ),
        )

        # ── Output filtering (keeps the field uncluttered) ──
        inputs.float(
            "min_score",
            default=0.0,
            label="Min score to keep",
            description="Only write candidates scoring at least this (0 = keep all).",
        )
        inputs.int(
            "max_per_image",
            default=0,
            label="Max candidates per image",
            description="Keep only the top-N highest-scoring per image (0 = unlimited).",
        )

        # ── Optional delegation ──
        inputs.bool(
            "delegate",
            default=False,
            label="Delegate execution",
            description=(
                "Run as a background (delegated) job — recommended for large datasets. "
                "Requires a running delegated service (fiftyone delegated launch). "
                "Leave off to run immediately."
            ),
            view=types.CheckboxView(),
        )

        return types.Property(
            inputs, view=types.View(label="Find Plane")
        )

    def resolve_delegation(self, ctx):
        # Optional delegation, driven by the checkbox in the form
        return bool(ctx.params.get("delegate", False))

    def _params(self, ctx):
        p = dict(DEFAULTS)
        for key in DEFAULTS:
            if ctx.params.get(key, None) is not None:
                p[key] = ctx.params[key]
        return p

    def execute(self, ctx):
        field = ctx.params.get("label_field") or "plane_candidates"
        p = self._params(ctx)
        view = ctx.target_view("target")
        dataset = ctx.dataset
        score_field = f"max_{field}_score"
        min_score = float(ctx.params.get("min_score") or 0.0)
        max_per_image = int(ctx.params.get("max_per_image") or 0)

        # Declare the output fields up front so the schema is persisted before any
        # autosave writes. Otherwise a later failure can leave field VALUES on the
        # sample documents without a corresponding schema entry, corrupting the
        # dataset (orphaned fields break hard reloads).
        schema = dataset.get_field_schema()
        if field not in schema:
            dataset.add_sample_field(
                field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections
            )
        if score_field not in schema:
            dataset.add_sample_field(score_field, fo.FloatField)

        total = len(view)
        n_flagged = 0
        n_with_hits = 0

        for i, sample in enumerate(view.iter_samples(autosave=True, progress=True)):
            img = cv2.imread(sample.filepath)
            if img is None:
                sample[field] = fo.Detections(detections=[])
                sample[score_field] = 0.0
                continue

            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cands = detect_candidates(gray, p)
            cands = _filter_and_cap(cands, min_score, max_per_image)

            sample[field] = _candidates_to_detections(cands, width, height)
            sample[score_field] = max((c["score"] for c in cands), default=0.0)

            n_flagged += len(cands)
            if cands:
                n_with_hits += 1

            if not ctx.delegated and total:
                ctx.set_progress(
                    progress=(i + 1) / total,
                    label=f"{i + 1}/{total} — {n_flagged} candidates",
                )

        # Declare the per-detection dynamic attributes (bbox_diag, rect_aspect, ...)
        # so they're filterable in the App. Best-effort: the detections are already
        # saved, so a hiccup here must never fail the whole run.
        try:
            dataset.add_dynamic_sample_fields()
        except Exception:
            pass
        dataset.save()

        return {
            "field": field,
            "score_field": score_field,
            "num_samples": total,
            "num_candidates": n_flagged,
            "num_samples_with_candidates": n_with_hits,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("num_samples", label="Samples processed")
        outputs.int("num_candidates", label="Candidate detections written")
        outputs.int("num_samples_with_candidates", label="Samples with ≥1 candidate")
        outputs.str("field", label="Detections field")
        outputs.str("score_field", label="Ranking field (sort by this, descending)")
        return types.Property(outputs, view=types.View(label="Find Plane — results"))

    def __call__(
        self,
        sample_collection,
        label_field="plane_candidates",
        dark_threshold=DEFAULTS["dark_threshold"],
        min_aspect_ratio=DEFAULTS["min_aspect_ratio"],
        max_interior_std=DEFAULTS["max_interior_std"],
        min_rect_fill=DEFAULTS["min_rect_fill"],
        min_diag_px=DEFAULTS["min_diag_px"],
        max_diag_px=DEFAULTS["max_diag_px"],
        min_area_px=DEFAULTS["min_area_px"],
        morph_kernel=DEFAULTS["morph_kernel"],
        min_score=0.0,
        max_per_image=0,
        delegate=False,
    ):
        """Run the detector from the SDK, e.g.::

            import fiftyone.operators as foo
            find_plane = foo.get_operator("@harpreetsahota/plane_finder/find_plane")
            find_plane(dataset, min_score=0.55, max_per_image=5, delegate=True)
        """
        params = dict(
            label_field=label_field,
            dark_threshold=dark_threshold,
            min_aspect_ratio=min_aspect_ratio,
            max_interior_std=max_interior_std,
            min_rect_fill=min_rect_fill,
            min_diag_px=min_diag_px,
            max_diag_px=max_diag_px,
            min_area_px=min_area_px,
            morph_kernel=morph_kernel,
            min_score=min_score,
            max_per_image=max_per_image,
            delegate=delegate,
        )
        return _execute(self.uri, sample_collection, params)


class PreviewDarkMask(foo.Operator):
    """Calibration helper: writes the thresholded dark mask as a fo.Heatmap so you
    can tune ``dark_threshold`` visually in the App. App-native replacement for the
    original ``--debug`` mask dump."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="preview_dark_mask",
            label="Preview Dark Mask (calibrate threshold)",
            description="Overlay the dark/morph mask as a heatmap to tune dark_threshold",
            dynamic=True,
            icon="/assets/icon.svg",
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        target_choices = types.RadioGroup()
        target_choices.add_choice("DATASET", label="Entire dataset")
        if ctx.view is not None and ctx.view != ctx.dataset.view():
            target_choices.add_choice("CURRENT_VIEW", label="Current view")
        if ctx.selected:
            target_choices.add_choice("SELECTED_SAMPLES", label="Selected samples")
        default_target = "SELECTED_SAMPLES" if ctx.selected else (
            "CURRENT_VIEW" if ctx.view is not None else "DATASET"
        )
        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            view=target_choices,
            label="Run on",
            description="Which samples to preview: the whole dataset, the current App view, or just your selected samples.",
        )

        inputs.int(
            "num_samples",
            default=10,
            label="Max samples to preview",
            description="Keep this small — calibration is for eyeballing, not the full run.",
        )
        inputs.str(
            "mask_field",
            default="dark_mask",
            required=True,
            label="Heatmap field",
            description="Sample field the heatmap is written to. Toggle it on in the App to view the mask overlay.",
        )
        inputs.int(
            "dark_threshold",
            default=DEFAULTS["dark_threshold"],
            label="Dark threshold (0-255)",
            description=(
                "The threshold being calibrated: pixels darker than this become mask. "
                "Re-run with different values until the mask cleanly isolates dark objects."
            ),
        )
        inputs.int(
            "morph_kernel",
            default=DEFAULTS["morph_kernel"],
            label="Morph close kernel",
            description="Size (px) of the morphological-close kernel, matching the find_plane setting you intend to use.",
        )
        inputs.bool(
            "apply_morph",
            default=True,
            label="Show post-morph mask",
            description="On: the mask that's actually contoured. Off: the raw threshold.",
            view=types.CheckboxView(),
        )
        inputs.str(
            "output_dir",
            default="",
            label="Mask output dir (optional)",
            description="Where mask PNGs are written. Blank = a sibling 'plane_finder_masks' folder.",
        )
        inputs.bool(
            "delegate",
            default=False,
            label="Delegate execution",
            description=(
                "Run as a background (delegated) job instead of immediately. Requires a "
                "running delegated service (fiftyone delegated launch). Usually unnecessary "
                "for a small calibration preview."
            ),
            view=types.CheckboxView(),
        )

        return types.Property(inputs, view=types.View(label="Preview Dark Mask"))

    def resolve_delegation(self, ctx):
        return bool(ctx.params.get("delegate", False))

    def execute(self, ctx):
        import os

        mask_field = ctx.params.get("mask_field") or "dark_mask"
        thresh = int(ctx.params.get("dark_threshold", DEFAULTS["dark_threshold"]))
        k = max(1, int(ctx.params.get("morph_kernel", DEFAULTS["morph_kernel"])))
        apply_morph = bool(ctx.params.get("apply_morph", True))
        num_samples = int(ctx.params.get("num_samples") or 10)

        view = ctx.target_view("target")
        if num_samples > 0:
            view = view.limit(num_samples)

        # Declare the heatmap field up front so autosave never writes an undeclared
        # field value (which would corrupt the dataset on a later failure).
        if mask_field not in ctx.dataset.get_field_schema():
            ctx.dataset.add_sample_field(
                mask_field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Heatmap
            )

        out_dir = ctx.params.get("output_dir") or ""

        n = 0
        for sample in view.iter_samples(autosave=True, progress=True):
            img = cv2.imread(sample.filepath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            if apply_morph:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Downscale heavy full-res masks to keep the heatmap light
            h, w = mask.shape[:2]
            scale = min(1.0, 1600 / max(h, w))
            if scale < 1.0:
                mask = cv2.resize(
                    mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST
                )

            sample_dir = out_dir or os.path.join(
                os.path.dirname(os.path.dirname(sample.filepath)),
                "plane_finder_masks",
            )
            os.makedirs(sample_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(sample.filepath))[0]
            mask_path = os.path.join(sample_dir, f"{stem}_t{thresh}.png")
            cv2.imwrite(mask_path, mask)

            sample[mask_field] = fo.Heatmap(map_path=mask_path)
            n += 1

        ctx.dataset.save()
        return {"mask_field": mask_field, "num_previewed": n, "dark_threshold": thresh}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("num_previewed", label="Samples previewed")
        outputs.int("dark_threshold", label="Threshold used")
        outputs.str("mask_field", label="Heatmap field (toggle it on in the App)")
        return types.Property(outputs, view=types.View(label="Preview Dark Mask — results"))

    def __call__(
        self,
        sample_collection,
        num_samples=10,
        mask_field="dark_mask",
        dark_threshold=DEFAULTS["dark_threshold"],
        morph_kernel=DEFAULTS["morph_kernel"],
        apply_morph=True,
        output_dir="",
        delegate=False,
    ):
        """Run the calibration preview from the SDK, e.g.::

            import fiftyone.operators as foo
            preview = foo.get_operator("@harpreetsahota/plane_finder/preview_dark_mask")
            preview(dataset, num_samples=10, dark_threshold=105)
        """
        params = dict(
            num_samples=num_samples,
            mask_field=mask_field,
            dark_threshold=dark_threshold,
            morph_kernel=morph_kernel,
            apply_morph=apply_morph,
            output_dir=output_dir,
            delegate=delegate,
        )
        return _execute(self.uri, sample_collection, params)


def register(p):
    p.register(FindPlane)
    p.register(PreviewDarkMask)

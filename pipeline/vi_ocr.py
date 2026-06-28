"""Vietnamese-side OCR from page images — Surya recogniser.

Why this exists
---------------
The translation pages carry a noisy embedded Tesseract text layer: titles come
out as "MINH MỆNH EHÍNH VẾU", "0UẾC-SỬ". That wrecks the Hán↔Việt alignment in
step 4. Re-OCRing the rendered page images recovers the diacritics Tesseract
drops.

Surya is a modern multilingual transformer OCR that does its OWN line detection
+ recognition. It recovers the tone marks and the b/h ascender that flatten on
this 1970s reprint ("tbần"->"thần", "Trằm"->"Trẫm"), at the source. Because Surya
detects its own lines, no separate PaddleOCR detector is needed here (and Surya
conflicts with Paddle over Pillow — run it in the clean Surya env, notebook ①).

No external API is used: Surya downloads pretrained weights once and runs
locally (GPU on Colab). Mirrors the no-API/Colab pipeline.

Public API
----------
    eng = VietnameseOCR(cfg, dpi)
    page_text: dict[int, str] = eng.ocr_pages(pdf_path, vi_pages)
"""
from __future__ import annotations

import re
from pathlib import Path

from . import config
from .common import get_logger

log = get_logger("vi_ocr")

# Surya's recognition emits inline formatting markup (<b>…</b>, <i>, <sup>,
# <math>, …) around bold titles / headers. It is noise for the Hán↔Việt
# alignment, so strip any such tag from each recognised line. This corpus has no
# legitimate angle-bracket content, so a blanket tag strip is safe.
_SURYA_TAG = re.compile(r"</?[^>]+>")


def _header_crop_top(arr, cfg) -> int:
    """Find the y to crop a page's top at, removing the running-head / page-number
    band. Returns 0 (no crop) unless a header band is confidently detected.

    Method: a horizontal projection profile (per-row ink count) groups rows into
    text *bands*. The top band is treated as a header — and cropped away — only
    when all of:
      * it starts within `header_zone` of the page height (top margin), and
      * it is a single thin line (height <= `header_max_lines` * median band), and
      * the gap to the next band is >= `header_gap_factor` * median line gap
        (and larger than half a line) — the wide whitespace under a running head.
    A multi-line content title or a chapter-opening page (no head, title set low)
    fails these tests and is left untouched. Validated on vol1 pages 20/45/82/…:
    strips "QUYỂN n  <page>" / lone page numbers, keeps titles + body.
    """
    import numpy as np

    H, W = arr.shape[:2]
    gray = arr if arr.ndim == 2 else arr[:, :, :3].mean(axis=2)
    ink = (gray < 128)
    row_ink = ink.sum(axis=1)
    thr = 0.012 * W                       # row is "text" if >~1.2% of width is ink
    min_gap = max(4, int(0.004 * H))      # merge intra-line breaks (descenders)
    on = row_ink > thr

    bands = []                            # [start, end] inclusive
    i = 0
    while i < H:
        if on[i]:
            j = i
            while j < H and on[j]:
                j += 1
            bands.append([i, j - 1])
            i = j
        else:
            i += 1
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= min_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    if len(merged) < 2:
        return 0

    heights = sorted(b[1] - b[0] + 1 for b in merged)
    gaps = sorted(merged[k + 1][0] - merged[k][1] for k in range(len(merged) - 1))
    med_h = heights[len(heights) // 2]
    med_gap = gaps[len(gaps) // 2]
    b0 = merged[0]
    b0_h = b0[1] - b0[0] + 1
    gap0 = merged[1][0] - b0[1]
    if (b0[0] < cfg.get("header_zone", 0.18) * H
            and b0_h <= cfg.get("header_max_lines", 1.8) * med_h
            and gap0 >= cfg.get("header_gap_factor", 1.4) * med_gap
            and gap0 > 0.5 * med_h):
        return max(0, merged[1][0] - int(0.4 * med_gap))   # crop just above body
    return 0


def _build_surya():
    """Surya detection + recognition predictors.

    Pin surya-ocr to the **det_predictor era (>=0.13,<0.16)**: pure-torch GPU,
    no Docker. Newer 0.20 moved full-page OCR behind a Docker service ("docker
    binary not found") and dropped the `det_predictor` kwarg — unusable on Colab.
    Usage: `recognition_predictor([img], det_predictor=detection_predictor)`.
    """
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
    except ImportError as e:  # pragma: no cover - guidance on missing dep
        raise ImportError(
            "VI image OCR needs surya-ocr 0.13-0.15: "
            'pip install "surya-ocr>=0.13,<0.16"'
        ) from e
    log.info("Surya OCR: building detection + recognition predictors")
    return DetectionPredictor(), RecognitionPredictor()


def _surya_lines(det, rec, pil_img):
    """Run Surya on one page image; return [(bbox, text), ...].

    Primary call is the 0.13-0.15 signature `rec([img], det_predictor=det)`; a
    `full_page=True` fallback covers a minor signature drift, but 0.20's Docker
    path is intentionally not supported (pin the version instead).
    """
    try:
        preds = rec([pil_img], det_predictor=det)
    except TypeError:                         # pragma: no cover - signature drift
        preds = rec([pil_img], full_page=True)
    page = preds[0]
    lines = getattr(page, "text_lines", None)
    if lines is None and not getattr(_surya_lines, "_warned", False):
        log.warning("Surya result has no .text_lines; attrs=%s",
                    [a for a in dir(page) if not a.startswith("_")])
        _surya_lines._warned = True
    out = []
    for ln in lines or []:
        txt = _SURYA_TAG.sub("", getattr(ln, "text", "") or "").strip()
        bbox = getattr(ln, "bbox", None)
        if txt and bbox and len(bbox) == 4:
            out.append((tuple(float(c) for c in bbox), txt))
    return out


def _reading_order(boxes, row_tol: float):
    """Sort line boxes top→bottom, left→right. Boxes whose y-centres are within
    `row_tol` px are treated as the same row (handles slight baseline wobble)."""
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    rows: list[list] = []
    for b in boxes:
        yc = (b[1] + b[3]) / 2
        if rows and abs(yc - (rows[-1][0][1] + rows[-1][0][3]) / 2) <= row_tol:
            rows[-1].append(b)
        else:
            rows.append([b])
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


class VietnameseOCR:
    def __init__(self, cfg: dict, dpi: int):
        self.cfg = cfg
        self.dpi = dpi
        self.surya_det, self.surya_rec = _build_surya()

    def _render(self, doc, page_no: int, save_dir: Path | None):
        """Rasterise one page of an already-open PDF to a numpy RGB array
        (and optionally cache the PNG)."""
        import fitz
        import numpy as np

        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = doc[page_no].get_pixmap(matrix=mat)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:                       # RGBA -> RGB
            arr = arr[:, :, :3]
        arr = np.ascontiguousarray(arr)
        if self.cfg.get("crop_header", True):   # drop running-head / page-number band
            y = _header_crop_top(arr, self.cfg)
            if y > 0:
                arr = np.ascontiguousarray(arr[y:, :])
        if save_dir is not None:             # cache the SAME image we OCR (post-crop)
            save_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(arr).save(save_dir / f"page_{page_no:04d}.png")
        return arr

    def page_lines(self, img) -> list[tuple[tuple[float, float, float, float], str]]:
        """Ordered [(bbox, text)] for a page (reading order, min-height filtered).

        bbox is (x0, y0, x1, y1) in the coordinate frame of the SAME image that
        was OCR'd — i.e. the post-header-crop array cached as pages_vi/*.png — so
        the boxes overlay correctly on that PNG (see pipeline/draw_boxes.py)."""
        return self._lines_surya(img)

    def ocr_page(self, img) -> str:
        return "\n".join(t for _, t in self.page_lines(img))

    def _lines_surya(self, img):
        from PIL import Image

        pil = Image.fromarray(img)
        try:
            lines = _surya_lines(self.surya_det, self.surya_rec, pil)
        except Exception as e:               # pragma: no cover - per-page robustness
            log.warning("Surya failed on a page (%s)", e)
            return []
        if not lines:
            return []
        min_h = self.cfg.get("min_line_height", 8)
        lines = [(b, t) for b, t in lines if (b[3] - b[1]) >= min_h]
        order = _reading_order([b for b, _ in lines], self.cfg.get("row_tol", 12))
        text_by_box = {b: t for b, t in lines}
        return [(b, text_by_box[b]) for b in order if b in text_by_box]

    def ocr_pages(self, pdf_path: Path, vi_pages: list[int],
                  save_dir: Path | None = None,
                  box_sink: dict | None = None) -> dict[int, str]:
        """OCR every VI page, logging per-page progress (page no, lines, chars)
        so a long run is visible. Uses a tqdm bar when available.

        If `box_sink` is given, it is filled `box_sink[page] = [(bbox, text), ...]`
        in reading order so a caller can persist per-line coordinates."""
        import fitz

        try:
            from tqdm.auto import tqdm
        except Exception:                       # pragma: no cover - tqdm optional
            tqdm = None

        out: dict[int, str] = {}
        total = len(vi_pages)
        n_chars = 0
        doc = fitz.open(pdf_path)               # open once, not per page
        bar = tqdm(total=total, unit="pg", desc="VI OCR (surya)") if tqdm else None
        try:
            for i, p in enumerate(vi_pages, 1):
                img = self._render(doc, p, save_dir)
                lines = self.page_lines(img)
                text = "\n".join(t for _, t in lines)
                out[p] = text
                if box_sink is not None:
                    box_sink[p] = lines
                lines = text.count("\n") + 1 if text else 0
                n_chars += len(text)
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(page=p, lines=lines, chars=n_chars)
                else:
                    log.info("OCR VI page %d/%d (pdf p%d): %d lines, %d chars total",
                             i, total, p, lines, n_chars)
        finally:
            if bar is not None:
                bar.close()
            doc.close()
        log.info("OCR done: %d/%d pages, %d chars total", total, total, n_chars)
        return out

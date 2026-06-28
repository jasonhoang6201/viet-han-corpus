"""Step 3 — Hán / SinoNom side: OCR -> reading order -> segmentation.

Input : out/<vol>/pages_han/*.png   (rendered by step 1)
Output: out/<vol>/han_boxes.jsonl       one row per detected box (column)
            {chapter, page, box_idx, id, bbox, sinonom, am_han_viet}
        out/<vol>/han_sentences.jsonl    one row per Hán sentence
            {chapter, page, sent_idx, id, sinonom, am_han_viet, box_ids}

Design notes
------------
* Classical Hán-Nôm is written in vertical columns, right-to-left, top-to-bottom
  (per the course guide). PaddleOCR's recognition model is trained on *horizontal*
  text, so it reads a vertical column almost randomly. Instead of detecting every
  column and rotating each crop, we rotate the WHOLE PAGE 90° counter-clockwise
  once: each vertical column becomes a horizontal line, the top-right (first)
  character lands top-left, and standard top→bottom / left→right reading order in
  the rotated frame *is* the classical reading order. So a single PaddleOCR call
  on the rotated page gives both good recognition and the right order.
  Detection boxes are mapped back to original-page coordinates for the Excel
  "Image box" column.
* âm Hán-Việt for every character comes from assets/dicts/hanviet.csv (Unihan).
* Sentence segmentation: woodblock prints are usually *unpunctuated*. If a page
  carries enough sentence-final punctuation we split on it; otherwise we treat
  each column/box as a sentence unit (the natural granularity for alignment).

Requires PaddleOCR 3.x (uses the `predict()` API and OCRResult format).

Run:  python -m pipeline.step3_sinonom --vol vol1
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

from . import config
from .common import (cjk_chars, get_logger, is_cjk, make_id, paddle_device,
                     progress, read_json, write_jsonl)

log = get_logger("step3")


# --------------------------------------------------------------------------- #
# Hán-Việt lookup
# --------------------------------------------------------------------------- #
def load_hanviet() -> dict[str, str]:
    table: dict[str, str] = {}
    if not config.HANVIET_CSV.exists():
        log.warning("hanviet.csv missing — run pipeline.build_dicts first")
        return table
    with config.HANVIET_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[row["sinonom_char"]] = row["am_han_viet"]
    return table


def transliterate(chars: str, table: dict[str, str]) -> str:
    return " ".join(table.get(c, "?") for c in chars if is_cjk(c))


# --------------------------------------------------------------------------- #
# OCR — PaddleOCR 3.x, whole-page rotation
# --------------------------------------------------------------------------- #
def _make_ocr(lang: str):
    """Build a PaddleOCR 3.x instance, GPU-first.

    On GPU (paddle GPU build, e.g. paddlepaddle-gpu 3.3.1/cu129) we run the
    **server** models (`PP-OCRv5_server_*`) — more accurate than mobile and fast
    on GPU — with `device="gpu"`. On a CPU box we fall back to **mobile** models
    plus `enable_mkldnn=True` (oneDNN conv): faster, and avoids the reference
    Im2Col conv path that segfaults old paddle CPU builds.
    We rotate the page ourselves, so the doc-orientation / unwarping / textline-
    orientation modules are disabled (download + latency only). Each kwarg set is
    tried in turn so the call still succeeds if a name drifts between 3.x builds.
    """
    from paddleocr import PaddleOCR

    device = paddle_device()
    base = dict(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if device == "gpu":
        attempts = [
            # preferred: server models on GPU (best accuracy, GPU is fast enough)
            dict(base, device="gpu",
                 text_detection_model_name="PP-OCRv5_server_det",
                 text_recognition_model_name="PP-OCRv5_server_rec"),
            dict(base, device="gpu"),      # default models on GPU
            base,                          # let paddle pick the device
            dict(lang=lang),               # minimal
        ]
    else:
        attempts = [
            # CPU: mobile models + oneDNN conv (fast, dodges the Im2Col segfault)
            dict(base, enable_mkldnn=True,
                 text_detection_model_name="PP-OCRv5_mobile_det",
                 text_recognition_model_name="PP-OCRv5_mobile_rec"),
            dict(base, enable_mkldnn=True),
            base,
            dict(lang=lang),
        ]
    log.info("PaddleOCR target device: %s", device)
    for i, kwargs in enumerate(attempts):
        try:
            ocr = PaddleOCR(**kwargs)
            extras = sorted(set(kwargs) - {"lang"})
            log.info("PaddleOCR built with: %s", extras or "lang only")
            return ocr
        except (ValueError, TypeError) as e:  # pragma: no cover - version drift
            if i == len(attempts) - 1:
                raise
            log.warning("PaddleOCR rejected %s (%s); trying a simpler ctor",
                        sorted(set(kwargs) - set(base)) or "base", e)


def _predict(ocr, img) -> list[tuple]:
    """Run PaddleOCR 3.x predict and return [(poly, text, score), ...].

    Defensive about the OCRResult container (dict-like in 3.x) and the exact
    polygon key (`rec_polys` aligns with `rec_texts`; falls back to `dt_polys`).
    """
    result = ocr.predict(img)
    if not result:
        return []
    res = result[0]

    def get(key):
        try:
            return res[key]
        except (KeyError, TypeError):
            return getattr(res, key, None)

    texts = get("rec_texts") or []
    scores = get("rec_scores") or []
    polys = get("rec_polys")
    if polys is None:
        polys = get("dt_polys")
    if polys is None:
        return []
    return list(zip(polys, texts, scores))


def _bbox_from_poly(poly) -> tuple[float, float, float, float]:
    xs = [float(pt[0]) for pt in poly]
    ys = [float(pt[1]) for pt in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _unrotate_bbox(rx0, ry0, rx1, ry1, orig_w: int) -> list[int]:
    """Map an axis-aligned bbox from the 90°-CCW-rotated frame back to original.

    Forward (cv2.ROTATE_90_COUNTERCLOCKWISE): original (x, y) -> (y, W-1-x).
    Inverse: rotated (rx, ry) -> original (W-1-ry, rx).
    """
    x0 = orig_w - 1 - ry1
    x1 = orig_w - 1 - ry0
    y0 = rx0
    y1 = rx1
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def _reading_order(boxes: list[dict]) -> list[dict]:
    """Order boxes by classical reading order using rotated-frame coordinates.

    In the rotated frame each column is a horizontal line; reading order is
    top→bottom (line) then left→right (within line). Each box carries a temporary
    `_rot` = (ry0, ry1, rx0) used only for sorting, stripped before returning.
    """
    if not boxes:
        return boxes
    heights = [b["_rot"][1] - b["_rot"][0] for b in boxes]
    band = (statistics.median(heights) or 1.0) * 0.6
    boxes.sort(key=lambda b: (round(((b["_rot"][0] + b["_rot"][1]) / 2) / band),
                              b["_rot"][2]))
    for b in boxes:
        del b["_rot"]
    return boxes


class HanOCR:
    """Single PaddleOCR 3.x instance; recognises a whole rotated page at once."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._ocr = _make_ocr(cfg["paddle_lang"])

    def page(self, img) -> list[dict]:
        import cv2

        # vertical RTL columns -> horizontal LTR lines (see module docstring)
        rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        orig_w = img.shape[1]

        boxes = []
        for poly, text, score in _predict(self._ocr, rot):
            chars = "".join(cjk_chars(text))
            if not chars:
                continue
            rx0, ry0, rx1, ry1 = _bbox_from_poly(poly)
            boxes.append({
                "bbox": _unrotate_bbox(rx0, ry0, rx1, ry1, orig_w),
                "sinonom": chars,
                "conf": round(float(score), 3),
                "_rot": (ry0, ry1, rx0),
            })
        return _reading_order(boxes)


def ocr_page(han_ocr: HanOCR, img_path: Path, cfg: dict) -> list[dict]:
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return []
    if cfg.get("preprocess", True):
        from .preprocess import preprocess_array
        pre = preprocess_array(img)
        # PaddleOCR expects 3-channel input
        img = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR) if pre.ndim == 2 else pre
    return han_ocr.page(img)


# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #
def segment_page(boxes: list[dict], cfg: dict) -> list[dict]:
    """Return sentence dicts: {sinonom, box_ids}. box_ids index into `boxes`."""
    full = "".join(b["sinonom"] for b in boxes)
    sent_marks = set(cfg["sentence_punct"])
    n_marks = sum(1 for c in full if c in sent_marks)
    n_cjk = sum(1 for c in full if is_cjk(c))

    if n_cjk and n_marks / max(n_cjk, 1) >= cfg["min_punct_ratio"]:
        # Punctuated: split the concatenated reading-ordered text on sentence marks.
        sents = []
        cur = []
        cur_boxes = set()
        for bi, b in enumerate(boxes):
            for ch in b["sinonom"]:
                cur.append(ch)
                cur_boxes.add(bi)
                if ch in sent_marks:
                    sents.append({"sinonom": "".join(cur), "box_ids": sorted(cur_boxes)})
                    cur, cur_boxes = [], set()
        if cur:
            sents.append({"sinonom": "".join(cur), "box_ids": sorted(cur_boxes)})
        return sents
    # Unpunctuated woodblock print: one sentence == one column/box.
    return [{"sinonom": b["sinonom"], "box_ids": [bi]} for bi, b in enumerate(boxes)]


def _clean(text: str, cfg: dict) -> str:
    """Drop clause/sentence punctuation for the pure-character column output."""
    drop = set(cfg["sentence_punct"]) | set(cfg["clause_punct"])
    return "".join(c for c in text if c not in drop)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _check_numpy() -> None:
    """Sanity-log the runtime stack. paddlepaddle-gpu 3.3.1 is built against
    numpy 2.x, so the old numpy-1.x pin is gone — only the legacy paddle 3.0.0
    CPU build had the numpy-2.x ABI segfault."""
    import numpy
    log.info("numpy %s, paddle device %s", numpy.__version__, paddle_device())


def run(vol: str, chapter: int = 1, limit: int | None = None) -> tuple[Path, Path]:
    _check_numpy()
    cfg = config.SINONOM
    out_dir = config.OUT_DIR / vol
    manifest = read_json(out_dir / "split_manifest.json")
    pages_dir = out_dir / "pages_han"
    img_paths = [pages_dir / f"page_{p:04d}.png" for p in manifest["han_pages"]]
    img_paths = [p for p in img_paths if p.exists()]
    if not img_paths:
        raise FileNotFoundError(f"no Hán page images in {pages_dir} — run step 1 with rendering")
    if limit:
        img_paths = img_paths[:limit]
        log.info("[%s] --limit %d: OCR only the first %d Hán pages", vol, limit, len(img_paths))

    hanviet = load_hanviet()
    log.info("[%s] OCR %d Hán pages (PaddleOCR/%s, whole-page rotate) ...",
             vol, len(img_paths), cfg["paddle_lang"])
    han_ocr = HanOCR(cfg)

    box_rows, sent_rows = [], []
    for img_path in progress(img_paths, "step3 Hán OCR", total=len(img_paths), log=log):
        page_no = int(re.search(r"(\d+)", img_path.stem).group(1))
        boxes = ocr_page(han_ocr, img_path, cfg)

        for b_idx, b in enumerate(boxes, start=1):
            chars = _clean(b["sinonom"], cfg)
            box_rows.append({
                "id": make_id(chapter, page_no, b_idx),
                "chapter": chapter, "page": page_no, "box_idx": b_idx,
                "bbox": b["bbox"], "sinonom": chars,
                "am_han_viet": transliterate(chars, hanviet),
                "conf": b["conf"],
            })

        for s_idx, s in enumerate(segment_page(boxes, cfg), start=1):
            chars = _clean(s["sinonom"], cfg)
            if not chars:
                continue
            sent_rows.append({
                "id": make_id(chapter, page_no, s_idx),
                "chapter": chapter, "page": page_no, "sent_idx": s_idx,
                "sinonom": chars,
                "am_han_viet": transliterate(chars, hanviet),
                "box_ids": s["box_ids"],
            })

    box_path = out_dir / "han_boxes.jsonl"
    sent_path = out_dir / "han_sentences.jsonl"
    write_jsonl(box_path, box_rows)
    write_jsonl(sent_path, sent_rows)
    log.info("[%s] wrote %d boxes, %d Hán sentences", vol, len(box_rows), len(sent_rows))
    return box_path, sent_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default="vol1")
    ap.add_argument("--chapter", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="OCR only the first N Hán pages (quick smoke test)")
    args = ap.parse_args()
    run(args.vol, args.chapter, limit=args.limit)


if __name__ == "__main__":
    main()

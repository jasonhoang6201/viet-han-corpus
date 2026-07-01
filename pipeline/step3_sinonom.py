"""Step 3 — Hán / SinoNom side: OCR -> reading order -> segmentation.

Input : out/<vol>/pages_han/*.png   (rendered by step 1)
Output: out/<vol>/han_boxes.jsonl       one row per COLUMN (merged, reading-ordered)
            {chapter, page, box_idx, id, bbox, sinonom, am_han_viet, conf, is_dbl}
        out/<vol>/han_sentences.jsonl    one row per Hán sentence
            {chapter, page, sent_idx, id, sinonom, am_han_viet, box_ids, is_dbl}

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
* PaddleOCR over-splits a column into stray boxes and mixes in the margin
  running-head + 雙行 interlinear notes. ``analyze_page`` clusters detections into
  columns by x-centre (RTL), merges vertical splits, drops the running-head band
  (both margins — it swaps side by page parity — plus the head-only bigram 政要卷之),
  and tags the hard 雙行 zones with ``is_dbl`` (geometry can't order them). Tuned +
  validated visually in notebooks/Minh_Menh_Han_Fragfix_TEST.ipynb.
* âm Hán-Việt for every character comes from assets/dicts/hanviet.csv (Unihan).
* Sentence segmentation: woodblock prints are usually *unpunctuated*. If a page
  carries enough sentence-final punctuation we split on it; otherwise we treat
  each column as a sentence unit (the natural granularity for alignment). ``is_dbl``
  rides through so a later real sentence-splitter / the aligner can isolate the
  approximate-order 雙行 text.

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
                     progress, read_jsonl, read_json, write_jsonl)

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


def _cx(b): return (b["bbox"][0] + b["bbox"][2]) / 2
def _cy(b): return (b["bbox"][1] + b["bbox"][3]) / 2
def _bw(b): return b["bbox"][2] - b["bbox"][0]
def _bh(b): return b["bbox"][3] - b["bbox"][1]
def _barea(b): return _bw(b) * _bh(b)


def analyze_page(boxes: list[dict], cfg: dict, page_w: int) -> list[dict]:
    """Turn raw PaddleOCR detections into reading-ordered COLUMNS.

    Woodblock pages are vertical columns, right-to-left. PaddleOCR over-splits a
    column into stray boxes and mixes in the margin running-head + 雙行 interlinear
    notes. We: drop tiny ink blobs, cluster detections into columns by x-centre
    (RTL), drop the running-head band (thin margin strip on EITHER side — it swaps
    margin by page parity — plus any cluster carrying the head-only bigram 政要卷之),
    merge true vertical splits (concatenate top→bottom), and tag the 雙行 zones
    (geometry can't order them reliably) with ``is_dbl`` so downstream isolates them.

    Returns ordered column dicts: {bbox (union), sinonom, conf, is_dbl}.
    Validated visually in notebooks/Minh_Menh_Han_Fragfix_TEST.ipynb.
    """
    idx = list(range(len(boxes)))
    if not boxes:
        return []
    wm = statistics.median([_bw(boxes[i]) for i in idx]) or 1.0
    am = statistics.median([_barea(boxes[i]) for i in idx]) or 1.0

    keep = []
    for i in idx:
        if cfg["drop_tiny"] and _barea(boxes[i]) < cfg["tiny_area_frac"] * am:
            continue                                    # ink blob / ▮ censored char
        keep.append(i)

    # cluster into columns by x-centre, right-to-left (reading order of columns)
    keep.sort(key=lambda i: -_cx(boxes[i]))
    clusters: list[list[int]] = []
    cur: list[int] = []
    for i in keep:
        if cur and abs(_cx(boxes[i]) - statistics.mean([_cx(boxes[j]) for j in cur])) \
                <= cfg["col_gap_frac"] * wm:
            cur.append(i)
        else:
            if cur:
                clusters.append(cur)
            cur = [i]
    if cur:
        clusters.append(cur)

    # running-head: (a) thin margin strip on either end, (b) any cluster carrying
    # the head-only bigram. Geometry is width-based only — a real date column like
    # 明命九年 is full-width so it is never dropped.
    if cfg["drop_running_head"] and clusters:
        def _is_head_strip(cl):
            cx = statistics.mean([_cx(boxes[i]) for i in cl])
            near = cx < cfg["head_margin_frac"] * page_w or \
                cx > page_w - cfg["head_margin_frac"] * page_w
            if not near:
                return False
            return statistics.mean([_bw(boxes[i]) for i in cl]) < cfg["head_narrow_frac"] * wm
        for end in (0, -1):
            if clusters and _is_head_strip(clusters[end]):
                clusters.pop(end)
        phrases = tuple(cfg["head_phrases"])
        clusters = [cl for cl in clusters
                    if not any(ph in "".join(boxes[i]["sinonom"] for i in cl) for ph in phrases)]

    columns = []
    for cl in clusters:
        cxs = sorted(cl, key=lambda i: -_cx(boxes[i]))
        span = _cx(boxes[cxs[0]]) - _cx(boxes[cxs[-1]])
        narrow = any(_bw(boxes[i]) < cfg["narrow_w_frac"] * wm for i in cl)
        is_dbl = len(cl) >= cfg["dbl_min_boxes"] or \
            (len(cl) >= 3 and narrow and span > cfg["dbl_split_frac"] * wm)
        if is_dbl:                                      # 雙行: guess right sub-col then left
            mid = (_cx(boxes[cxs[0]]) + _cx(boxes[cxs[-1]])) / 2
            right = sorted([i for i in cl if _cx(boxes[i]) >= mid], key=lambda i: _cy(boxes[i]))
            left = sorted([i for i in cl if _cx(boxes[i]) < mid], key=lambda i: _cy(boxes[i]))
            seq = right + left
        else:                                           # single column: top -> bottom
            seq = sorted(cl, key=lambda i: _cy(boxes[i]))
        xs = [boxes[i]["bbox"][0] for i in seq] + [boxes[i]["bbox"][2] for i in seq]
        ys = [boxes[i]["bbox"][1] for i in seq] + [boxes[i]["bbox"][3] for i in seq]
        columns.append({
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "sinonom": "".join(boxes[i]["sinonom"] for i in seq),
            "conf": round(min(boxes[i]["conf"] for i in seq), 3),
            "is_dbl": is_dbl,
        })
    return columns


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
            })
        # cluster raw detections -> merged, reading-ordered columns (+ head drop, 雙行 tag)
        return analyze_page(boxes, self.cfg, orig_w)


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
    """Return sentence dicts: {sinonom, box_ids, is_dbl}. box_ids index into `boxes`.

    `is_dbl` marks a sentence that came from a 雙行 interlinear zone whose reading
    order is only a geometric guess — downstream (sentence split / alignment) can
    isolate or down-weight it. Real column text carries is_dbl=False.
    """
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
                    ids = sorted(cur_boxes)
                    sents.append({"sinonom": "".join(cur), "box_ids": ids,
                                  "is_dbl": any(boxes[i].get("is_dbl") for i in ids)})
                    cur, cur_boxes = [], set()
        if cur:
            ids = sorted(cur_boxes)
            sents.append({"sinonom": "".join(cur), "box_ids": ids,
                          "is_dbl": any(boxes[i].get("is_dbl") for i in ids)})
        return sents
    # Unpunctuated woodblock print: one sentence == one column/box.
    return [{"sinonom": b["sinonom"], "box_ids": [bi], "is_dbl": bool(b.get("is_dbl"))}
            for bi, b in enumerate(boxes)]


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
                "conf": b["conf"], "is_dbl": bool(b.get("is_dbl")),
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
                # GLOBAL box ids (page-local index -> box id) so han_sentences is
                # consistent whether or not step 3c re-segments it in the notebook.
                "box_ids": [make_id(chapter, page_no, bi + 1) for bi in s["box_ids"]],
                "is_dbl": bool(s.get("is_dbl")),
            })

    box_path = out_dir / "han_boxes.jsonl"
    sent_path = out_dir / "han_sentences.jsonl"
    write_jsonl(box_path, box_rows)
    write_jsonl(sent_path, sent_rows)
    log.info("[%s] wrote %d boxes, %d Hán sentences", vol, len(box_rows), len(sent_rows))
    return box_path, sent_path


def review(vol: str) -> Path:
    """Hán review queue — low-confidence boxes.

    Moved out of step 4 so P3 is alignment-only. Run AFTER the Qwen consensus
    (step3b) has rewritten han_boxes.jsonl, so the confidences reflect the
    corrected chars. OCR correctness is judged by the consensus + Qwen arbiter,
    so the queue only surfaces low-confidence boxes. Reads han_boxes.jsonl;
    writes han_review.jsonl.
    """
    from .reviews import build_review

    out_dir = config.OUT_DIR / vol
    boxes = list(read_jsonl(out_dir / "han_boxes.jsonl"))

    review_rows = build_review(boxes, config.REVIEW)
    rev_path = out_dir / "han_review.jsonl"
    write_jsonl(rev_path, review_rows)
    log.info("[%s] Hán review: %d low-conf boxes -> %s",
             vol, len(review_rows), rev_path.name)
    return rev_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default="vol1")
    ap.add_argument("--chapter", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="OCR only the first N Hán pages (quick smoke test)")
    ap.add_argument("--review", action="store_true",
                    help="skip OCR; build han_review from existing han_boxes "
                         "(run after the Qwen consensus step)")
    args = ap.parse_args()
    if args.review:
        review(args.vol)
    else:
        run(args.vol, args.chapter, limit=args.limit)


if __name__ == "__main__":
    main()

"""Overlay OCR bounding boxes on the rendered pages for visual QA.

Two overlays, each written to its own folder so the originals stay clean:

  PLAIN  — every detected box, numbered in reading order. Lets you check that
           detection landed on real text columns / lines.
             out/<vol>/pages_han_boxed/page_NNNN.png   (red)
             out/<vol>/pages_vi_boxed/page_NNNN.png    (blue)

  ALIGNED — boxes coloured by their sentence-alignment pair (one colour per
            Hán↔Việt pair, same colour + index on both sides) so you can see
            which Hán columns map to which Vietnamese lines. Needs the aligner
            output out/<vol>/alignment.jsonl (notebook ②).
             out/<vol>/pages_han_aligned/page_NNNN.png
             out/<vol>/pages_vi_aligned/page_NNNN.png

Coordinate frames match each PNG (Hán: original rendered page, as make_report
crops from; VI: post-header-crop image actually OCR'd), so no rescaling.

Box→pair chain:
  * Hán: pair.han_ids -> han_sentences.box_ids (0-based page-local) -> +1 ->
         han_boxes (page, box_idx) -> bbox.
  * Việt: pair.vie_ids -> vi_sentences.box_ids (vi_boxes ids) -> bbox.

Run:  python -m pipeline.draw_boxes --vol vol1
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config
from .common import get_logger, read_jsonl

log = get_logger("draw_boxes")

# Distinct, high-contrast colours cycled across alignment pairs. Adjacent pairs
# get different hues; recycling is fine — proximity + index disambiguate.
_PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (166, 86, 40), (247, 129, 191), (153, 153, 153),
    (0, 158, 115), (213, 94, 0), (86, 180, 233), (240, 228, 66),
]


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:                    # pragma: no cover - font availability
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Plain per-box overlay
# --------------------------------------------------------------------------- #
def _draw_plain(img_path: Path, boxes: list[dict], out_path: Path, color) -> None:
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _font(max(14, im.height // 70))
    for b in boxes:
        bb = b.get("bbox")
        if not bb or len(bb) != 4:
            continue
        x0, y0, x1, y1 = (float(c) for c in bb)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = str(b.get("box_idx", "") or "")
        if label:
            draw.text((x0 + 2, y0 + 1), label, fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def _plain_layer(out_dir: Path, boxes_file: str, pages_sub: str, boxed_sub: str, color) -> int:
    bp = out_dir / boxes_file
    if not bp.exists():
        log.warning("%s missing — skip %s plain overlay", boxes_file, pages_sub)
        return 0
    by_page: dict[int, list[dict]] = defaultdict(list)
    for r in read_jsonl(bp):
        by_page[r["page"]].append(r)
    n = 0
    for page_no, boxes in sorted(by_page.items()):
        img_path = out_dir / pages_sub / f"page_{page_no:04d}.png"
        if not img_path.exists():
            log.warning("page image %s missing — skip", img_path.name)
            continue
        _draw_plain(img_path, boxes, out_dir / boxed_sub / f"page_{page_no:04d}.png", color)
        n += 1
    log.info("[%s] plain: drew boxes on %d pages -> %s", out_dir.name, n, boxed_sub)
    return n


# --------------------------------------------------------------------------- #
# Alignment-pair overlay
# --------------------------------------------------------------------------- #
def _draw_aligned(img_path: Path, items: list[tuple], out_path: Path) -> None:
    """items: [(bbox, color, label)] — coloured rectangle + pair index per box."""
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _font(max(14, im.height // 70))
    for bbox, color, label in items:
        x0, y0, x1, y1 = (float(c) for c in bbox)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 2, y0 + 1), str(label), fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def _aligned_layers(out_dir: Path, align_path: Path | None = None) -> int:
    align_p = align_path or (out_dir / "alignment.jsonl")
    if not align_p.exists():
        log.info("[%s] %s missing — skip aligned overlay (run notebook ② first, "
                 "or pass --align output_bge/<vol>/alignment.jsonl)",
                 out_dir.name, align_p.name)
        return 0
    pairs = list(read_jsonl(align_p))

    # Hán: (page, box_idx) -> bbox ; sentence id -> (page, 0-based box_ids)
    han_bbox = {(r["page"], r["box_idx"]): r["bbox"]
                for r in read_jsonl(out_dir / "han_boxes.jsonl")}
    han_sent = {r["id"]: r for r in read_jsonl(out_dir / "han_sentences.jsonl")}
    # Việt: vi_box id -> (page, bbox) ; sentence id -> row (carries box_ids)
    vi_box: dict[str, tuple[int, list]] = {}
    vi_path = out_dir / "vi_boxes.jsonl"
    if vi_path.exists():
        vi_box = {r["id"]: (r["page"], r["bbox"]) for r in read_jsonl(vi_path)}
    vi_sent_path = out_dir / "vi_sentences.jsonl"
    vi_sent = {r["id"]: r for r in read_jsonl(vi_sent_path)} if vi_sent_path.exists() else {}

    han_pages: dict[int, list[tuple]] = defaultdict(list)
    vi_pages: dict[int, list[tuple]] = defaultdict(list)
    for pi, p in enumerate(pairs, start=1):
        color = _PALETTE[pi % len(_PALETTE)]
        for hid in p.get("han_ids", []):
            row = han_sent.get(hid)
            if not row:
                continue
            for bi in row.get("box_ids", []):
                bbox = han_bbox.get((row["page"], bi + 1))   # box_ids are 0-based
                if bbox:
                    han_pages[row["page"]].append((bbox, color, pi))
        for vid in p.get("vie_ids", []):
            row = vi_sent.get(vid)
            if not row:
                continue
            for bxid in row.get("box_ids", []):
                hit = vi_box.get(bxid)
                if hit:
                    page, bbox = hit
                    vi_pages[page].append((bbox, color, pi))

    n = 0
    for page_no, items in sorted(han_pages.items()):
        img = out_dir / "pages_han" / f"page_{page_no:04d}.png"
        if img.exists():
            _draw_aligned(img, items, out_dir / "pages_han_aligned" / f"page_{page_no:04d}.png")
            n += 1
    for page_no, items in sorted(vi_pages.items()):
        img = out_dir / "pages_vi" / f"page_{page_no:04d}.png"
        if img.exists():
            _draw_aligned(img, items, out_dir / "pages_vi_aligned" / f"page_{page_no:04d}.png")
            n += 1
    log.info("[%s] aligned: %d pairs -> %d pages (pages_han_aligned, pages_vi_aligned)",
             out_dir.name, len(pairs), n)
    return n


def run(vol: str, align: Path | None = None) -> None:
    out_dir = config.OUT_DIR / vol
    _plain_layer(out_dir, "han_boxes.jsonl", "pages_han", "pages_han_boxed", (220, 30, 30))
    _plain_layer(out_dir, "vi_boxes.jsonl", "pages_vi", "pages_vi_boxed", (30, 110, 220))
    _aligned_layers(out_dir, align)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay OCR / alignment boxes on page images")
    ap.add_argument("--vol", default="vol1")
    ap.add_argument("--align", type=Path, default=None,
                    help="alignment.jsonl path (default out/<vol>/; "
                         "use output_bge/<vol>/alignment.jsonl for the bge run)")
    args = ap.parse_args()
    run(args.vol, align=args.align)


if __name__ == "__main__":
    main()

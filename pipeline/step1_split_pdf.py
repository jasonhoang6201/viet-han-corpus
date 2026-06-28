"""Step 1 — Split a scanned volume into its Vietnamese and Hán halves.

Minh Mệnh Chính Yếu volumes are scanned images with a noisy embedded OCR text
layer. The book layout is: front-matter + Vietnamese (Quốc ngữ) translation
first, then the Hán (classical Chinese) original. We must find that boundary.

Approach (matches the user's plan):
  1. For every page, read the embedded text layer and tokenise it.
  2. Classify each page:
       - BLANK : too few tokens (image-only / break page)
       - VI    : >= `vi_word_ratio` of tokens are real Vietnamese words
       - HAN   : otherwise (the Hán layer OCRs to non-Vietnamese garbage)
  3. The Hán section is confirmed only at the first HAN page that begins a run
     of MORE THAN `han_confirm_run` consecutive Hán pages (blank pages don't
     break the run but cannot start it). This rejects chapter-title / plate
     pages inside the Vietnamese half.
  4. Blank "break" pages are excluded from BOTH halves.

Outputs (out/<vol>/):
  * split_manifest.json   page classification + the chosen boundary
  * <vol>_vi.pdf          Vietnamese-half pages only
  * <vol>_han.pdf         Hán-half pages only
  * pages_han/*.png       rasterised Hán pages (for step 3)

Step 2 reads the embedded Vietnamese text layer directly (no OCR), so the VI
pages are not rasterised here — only the Hán pages, which step 3 must re-OCR.

Run:  python -m pipeline.step1_split_pdf --vol vol1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF

from . import config
from .common import get_logger, progress, vi_diacritic_ratio, write_json

log = get_logger("step1")

BLANK, VI, HAN, PLATE = "BLANK", "VI", "HAN", "PLATE"


def ink_ratio(page: "fitz.Page", dpi: int) -> float:
    """Fraction of dark (ink) pixels on the rasterised page."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    data = pix.samples
    if not data:
        return 0.0
    dark = sum(1 for b in data if b < 128)
    return dark / len(data)


def classify_page(page: "fitz.Page", cfg: dict) -> tuple[str, dict]:
    ntok, diac = vi_diacritic_ratio(page.get_text())
    if ntok < cfg["min_tokens"]:
        # No usable text: separate an image-only Hán plate (woodblock cover/title)
        # from a truly-blank separator page by ink coverage.
        ink = ink_ratio(page, cfg["plate_ink_dpi"])
        label = PLATE if ink >= cfg["plate_ink_ratio"] else BLANK
        return label, {"tokens": ntok, "diac_ratio": round(diac, 3), "ink": round(ink, 4)}
    label = VI if diac >= cfg["vi_diac_ratio"] else HAN
    return label, {"tokens": ntok, "diac_ratio": round(diac, 3)}


def find_han_start(labels: list[str], confirm_run: int) -> int | None:
    """Index of the first HAN page that starts a run of > confirm_run HAN pages.

    Blank and plate pages are skipped (do not break the run) but a run must
    *start* on a HAN page, never a blank/plate one.
    """
    n = len(labels)
    for i in range(n):
        if labels[i] != HAN:
            continue
        run, j = 0, i
        while j < n and labels[j] != VI:      # VI breaks the run; BLANK doesn't
            if labels[j] == HAN:
                run += 1
            j += 1
            if run > confirm_run:
                return i
    return None


def find_vi_body_start(doc: "fitz.Document", vi_pages: list[int], cfg: dict) -> int | None:
    """First VI page where the translation BODY begins (front matter ends).

    The body is the part-by-part rendering of the Hán original, and it cites the
    Hán leaf it came from in brackets — "[1a]", "[1b]", a bare "[1]", or the
    spelled form "[tờ 3b]". Cover / half-title / colophon / translator-credit
    pages never carry a leaf marker, so the first VI page that does (and reads as
    real body: dense, or naming a chapter / reign) marks the boundary. Everything
    before it is Vietnamese-only front matter with no Hán counterpart.

    Fallback when no marker OCRs cleanly: the last "… dịch của <NAME>" half-title
    page, with the body taken as the next dense page after it.

    Returns the page index (an element of vi_pages) or None to keep all VI pages.
    """
    if not cfg.get("trim_front_matter", True) or not vi_pages:
        return None
    leaf = re.compile(cfg["leaf_marker_regex"])
    kw = re.compile(cfg["body_keyword_regex"], re.I)
    halftitle = re.compile(cfg["halftitle_regex"], re.I)
    min_tok = cfg["body_min_tokens"]

    last_halftitle = None
    for p in vi_pages:
        text = doc[p].get_text()
        ntok = len(text.split())
        if leaf.search(text) and (ntok >= min_tok or kw.search(text)):
            return p
        if halftitle.search(text) and ntok < min_tok:   # short credit page
            last_halftitle = p

    if last_halftitle is not None:                       # fallback: after credit page
        for p in vi_pages:
            if p > last_halftitle and len(doc[p].get_text().split()) >= min_tok:
                return p
    return None


def split_volume(vol: str) -> dict:
    cfg = config.SPLIT
    pdf_path = config.DATA_DIR / f"{vol}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    out_dir = config.OUT_DIR / vol
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    pages = []
    labels = []
    for i in progress(range(len(doc)), "step1 classify pages", total=len(doc), log=log):
        label, meta = classify_page(doc[i], cfg)
        labels.append(label)
        pages.append({"page": i, "label": label, **meta})

    han_text_start = find_han_start(labels, cfg["han_confirm_run"])
    if han_text_start is None:
        log.warning("[%s] no confirmed Hán section found", vol)
        han_start = len(doc)
    else:
        # The confirmed Hán run starts on the first text-bearing Hán page, but
        # the real boundary is right after the last Vietnamese page: the gap in
        # between holds the Hán front matter (woodblock title/cover plates such
        # as 明命政要) that must stay with the Hán half.
        last_vi = max((p["page"] for p in pages
                       if p["page"] < han_text_start and p["label"] == VI),
                      default=han_text_start - 1)
        han_start = last_vi + 1

    # Vietnamese half: VI pages before the boundary. Hán half: every Hán text
    # page AND image-only plate from the boundary onward. Truly-blank separator
    # pages are dropped from both halves.
    vi_pages = [p["page"] for p in pages if p["page"] < han_start and p["label"] == VI]
    han_pages = [p["page"] for p in pages
                 if p["page"] >= han_start and p["label"] in (HAN, PLATE)]

    # Drop the Vietnamese-only front matter (cover/half-title/colophon) that has
    # no Hán counterpart and would force-match the Hán side in step 4.
    vi_body_start = find_vi_body_start(doc, vi_pages, cfg)
    vi_front_matter = []
    if vi_body_start is not None:
        vi_front_matter = [p for p in vi_pages if p < vi_body_start]
        vi_pages = [p for p in vi_pages if p >= vi_body_start]
        log.info("[%s] VI body starts at p%d | dropped %d front-matter page(s): %s",
                 vol, vi_body_start, len(vi_front_matter), vi_front_matter)
    else:
        log.warning("[%s] no VI body-start marker found — keeping all VI pages", vol)

    log.info("[%s] %d pages | Hán starts at p%d | VI=%d HAN=%d (blanks dropped)",
             vol, len(doc), han_start, len(vi_pages), len(han_pages))

    _export_pdf(doc, vi_pages, out_dir / f"{vol}_vi.pdf")
    _export_pdf(doc, han_pages, out_dir / f"{vol}_han.pdf")
    # VI pages are not rasterised: step 2 reads the embedded text layer directly.
    _render(doc, han_pages, out_dir / "pages_han", cfg["render_dpi"])

    manifest = {
        "volume": vol,
        "num_pages": len(doc),
        "han_start": han_start,
        "vi_body_start": vi_body_start,
        "vi_front_matter": vi_front_matter,
        "config": cfg,
        "vi_pages": vi_pages,
        "han_pages": han_pages,
        "pages": pages,
    }
    write_json(out_dir / "split_manifest.json", manifest)
    doc.close()
    return manifest


def _export_pdf(doc: "fitz.Document", page_indices: list[int], path: Path) -> None:
    if not page_indices:
        return
    new = fitz.open()
    for i in page_indices:
        new.insert_pdf(doc, from_page=i, to_page=i)
    new.save(path)
    new.close()
    log.info("wrote %s (%d pages)", path.name, len(page_indices))


def _render(doc: "fitz.Document", page_indices: list[int], out: Path, dpi: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i in progress(page_indices, f"step1 render {out.name}", total=len(page_indices), log=log):
        pix = doc[i].get_pixmap(matrix=mat)
        pix.save(out / f"page_{i:04d}.png")
    log.info("rendered %d images -> %s", len(page_indices), out.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default="vol1", help="volume stem, e.g. vol1")
    ap.add_argument("--no-render", action="store_true",
                    help="skip rasterising page images (faster, classification only)")
    args = ap.parse_args()
    if args.no_render:
        config.SPLIT["render_dpi"] = 0
        # monkeypatch _render to a no-op
        global _render
        _render = lambda *a, **k: log.info("(skipped rendering)")  # noqa: E731
    split_volume(args.vol)


if __name__ == "__main__":
    main()

"""Step 2 — Vietnamese (Quốc ngữ) side: read text -> spell-fix -> sentence split.

Input : data/<vol>.pdf           (the scanned VI pages, re-OCR'd with Surya)
        out/<vol>/split_manifest.json   (which pages are the Vietnamese half)
Output: out/<vol>/vi_sentences.jsonl
            {chapter, page, sent_idx, id, text, raw_text}

Design notes
------------
* Text comes from re-OCRing the rendered VI page images with Surya, which does
  its own line detection + recognition (pipeline/vi_ocr.py). The embedded
  Tesseract layer mangles diacritics ("MINH MỆNH EHÍNH VẾU"), which drags down
  step-4 alignment; Surya recovers them.
* Spell normalisation is dictionary-based and *conservative*: an out-of-vocab
  token is only rewritten when exactly one in-vocab candidate exists within a
  small edit distance, and the original is always preserved in `raw_text`.
* Sentence segmentation uses underthesea when available, with a regex fallback.

Run:  python -m pipeline.step2_vietnamese --vol vol1
"""
from __future__ import annotations

import argparse
import re
from functools import lru_cache

from . import config
from .clean_vi import CLEAN_RULES, _SPACE_PUNCT, _WS, clean_text, is_junk
from .common import get_logger, load_vi_vocab, make_id, oov_rate, read_json, write_jsonl

log = get_logger("step2")


# Vietnamese-aware "word character" class for correction-map boundaries.
_VI_WORD = r"0-9A-Za-zÀ-ỹĐđ"


# --------------------------------------------------------------------------- #
# Domain correction map (phrase-level, deterministic)
# --------------------------------------------------------------------------- #
class CorrectionMap:
    """Apply curated `wrong -> correct` fixes from vi_corrections.csv.

    Handles what the edit-distance SpellFixer can't: multi-syllable terms,
    hyphenated proper nouns and reign titles (e.g. 'Triệu-trị' -> 'Thiệu Trị').

    Matching rules:
      * case-insensitive; a space in `wrong` matches one or more spaces OR
        hyphens, so one row covers 'Triệu trị', 'Triệu-trị', 'Triệu  trị';
      * Vietnamese-aware word boundaries (won't fire mid-word);
      * the replacement copies the matched casing (ALL CAPS -> caps,
        Titlecase -> titlecase, else as written).
    """

    def __init__(self, path):
        self.rules: list[tuple] = []   # (compiled_regex, correct)
        if not path.exists():
            log.warning("correction map %s not found; skipping domain fixes", path)
            return
        import csv

        for row in csv.reader(path.read_text(encoding="utf-8").splitlines()):
            if not row or row[0].lstrip().startswith("#") or len(row) < 2:
                continue
            wrong, correct = row[0].strip(), row[1].strip()
            if not wrong or wrong.lower() == "wrong":   # skip header
                continue
            body = r"[\s\-]+".join(re.escape(t) for t in wrong.split())
            pat = re.compile(rf"(?<![{_VI_WORD}]){body}(?![{_VI_WORD}])",
                             flags=re.IGNORECASE | re.UNICODE)
            self.rules.append((pat, correct))
        log.info("correction map: %d rules from %s", len(self.rules), path.name)

    @staticmethod
    def _recase(matched: str, repl: str) -> str:
        if matched.isupper():
            return repl.upper()
        if matched[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    def apply(self, text: str) -> str:
        for pat, correct in self.rules:
            text = pat.sub(lambda m: self._recase(m.group(0), correct), text)
        return text


# Single-character OCR confusions on this 1970s typeset reprint. Each pair is
# bidirectional. The recogniser occasionally drops the `h` ascender after a
# stop ("theo"->"tbeo", "thần"->"tbần") or flattens a few vowel/tone glyphs.
# Applied per OOV token, accepted only when EXACTLY ONE substitution lands in
# vocab — so the rule self-limits (a confusion that yields a non-word does
# nothing). Keep this list SHORT and high-precision; tone-level errors that stay
# ambiguous are left for the recogniser (Surya) / human reviewer.
_CONFUSABLE_PAIRS = (
    ("b", "h"),   # tbần->thần, tbeo->theo  (dropped ascender)
    ("o", "ô"),   # so->sô, cong->công
    ("e", "ê"),   # ben->bên
    ("u", "ư"),   # tu->tư
    ("d", "đ"),   # dao->đao
)


# --------------------------------------------------------------------------- #
# Conservative dictionary-based spell normalisation
# --------------------------------------------------------------------------- #
class SpellFixer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.vocab = load_vi_vocab(single_syllable=True)
        # index vocab by length for fast edit-distance candidate generation
        self.by_len: dict[int, set[str]] = {}
        for w in self.vocab:
            self.by_len.setdefault(len(w), set()).add(w)

    def _confusable_fix(self, low: str) -> str | None:
        """Apply each known char confusion once; return the unique in-vocab
        result, or None if zero or several confusions land in vocab."""
        hits = set()
        for i, ch in enumerate(low):
            for a, b in _CONFUSABLE_PAIRS:
                for src, dst in ((a, b), (b, a)):
                    if ch == src:
                        cand = low[:i] + dst + low[i + 1:]
                        if cand in self.vocab:
                            hits.add(cand)
        return next(iter(hits)) if len(hits) == 1 else None

    @lru_cache(maxsize=100_000)
    def _candidates(self, token: str) -> tuple[str, ...]:
        """In-vocab words within edit distance `spell_max_edit_distance`."""
        maxd = self.cfg["spell_max_edit_distance"]
        cands = []
        for L in range(len(token) - maxd, len(token) + maxd + 1):
            for w in self.by_len.get(L, ()):  # noqa: E741
                if _edit_distance_le(token, w, maxd):
                    cands.append(w)
        return tuple(cands)

    def fix_token(self, token: str) -> str:
        low = token.lower()
        if (len(low) < self.cfg["spell_min_token_len"]
                or low in self.vocab or not low.isalpha()):
            return token
        hit = self._confusable_fix(low)          # targeted OCR confusions first
        if hit is not None:
            return _match_case(token, hit)
        cands = self._candidates(low)
        if len(cands) == 1:                      # unambiguous -> apply
            return _match_case(token, cands[0])
        return token

    def fix_text(self, text: str) -> str:
        return re.sub(r"[^\W\d_]+", lambda m: self.fix_token(m.group(0)), text,
                      flags=re.UNICODE)


def _edit_distance_le(a: str, b: str, maxd: int) -> bool:
    """True if Levenshtein(a, b) <= maxd (early-exit banded DP)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > maxd:
        return False
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        lo, hi = max(1, i - maxd), min(lb, i + maxd)
        best = maxd + 1
        for j in range(1, lb + 1):
            if j < lo or j > hi:
                cur[j] = maxd + 1
                continue
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            best = min(best, cur[j])
        if best > maxd:
            return False
        prev = cur
    return prev[lb] <= maxd


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl.capitalize()
    return repl


# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #
def get_sentence_splitter():
    try:
        from underthesea import sent_tokenize

        def split(text: str) -> list[str]:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        log.info("sentence splitter: underthesea")
        return split
    except Exception:  # pragma: no cover - fallback
        log.warning("underthesea unavailable; using regex sentence splitter")
        pat = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ỵ0-9])")

        def split(text: str) -> list[str]:
            return [s.strip() for s in pat.split(text) if s.strip()]
        return split


_NB_HYPHEN_NL = re.compile(r"-\s*\n\s*")
_NB_NL = re.compile(r"\s*\n\s*")
_NB_TAB = re.compile(r"[ \t]{2,}")


def normalise_block(text: str) -> str:
    """Collapse OCR line breaks into flowing text, de-hyphenate line wraps."""
    text = _NB_HYPHEN_NL.sub("", text)        # join hyphenated line breaks
    text = _NB_NL.sub(" ", text)              # other line breaks -> space
    text = _NB_TAB.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Char-level provenance: map each surviving block character back to the OCR box
# it came from, so a sentence (re-segmented across lines/pages) can list the
# detection boxes it covers. Replays the SAME normalise_block + clean_text
# transforms on a (char, box-id) stream; the produced text is asserted identical
# to the real cleaner, else we fall back to no box_ids for that page.
# --------------------------------------------------------------------------- #
def _sub_const(pat, repl: str, s: str, prov: list):
    """re.sub with a constant replacement, carrying per-char provenance.
    Inserted replacement chars inherit the provenance of the match's first char."""
    out_s, out_p, last = [], [], 0
    for m in pat.finditer(s):
        a, b = m.start(), m.end()
        out_s.append(s[last:a]); out_p.extend(prov[last:a])
        src = prov[a] if a < b else None
        out_s.append(repl); out_p.extend([src] * len(repl))
        last = b
    out_s.append(s[last:]); out_p.extend(prov[last:])
    return "".join(out_s), out_p


def _sub_group1(pat, s: str, prov: list):
    """re.sub(pat, r"\1", ...) variant — keeps group(1) with its own provenance."""
    out_s, out_p, last = [], [], 0
    for m in pat.finditer(s):
        out_s.append(s[last:m.start()]); out_p.extend(prov[last:m.start()])
        gi, g = m.start(1), m.group(1)
        out_s.append(g); out_p.extend(prov[gi:gi + len(g)])
        last = m.end()
    out_s.append(s[last:]); out_p.extend(prov[last:])
    return "".join(out_s), out_p


def _strip_prov(s: str, prov: list):
    lead = len(s) - len(s.lstrip())
    s2 = s.strip()
    return s2, prov[lead:lead + len(s2)]


def block_with_prov(box_texts: list[str], page_no: int):
    """Return (block_text, prov) where prov[i] is the (page, box_idx) that block
    char i came from (box_idx 1-based, matching vi_boxes.jsonl). Returns
    (None, None) if the provenance replay diverges from the real cleaner."""
    chars, prov = [], []
    for b_idx, t in enumerate(box_texts, start=1):
        if b_idx > 1:                          # ocr_pages joins lines with "\n"
            chars.append("\n"); prov.append(None)
        chars.extend(t); prov.extend([(page_no, b_idx)] * len(t))
    s = "".join(chars)
    # normalise_block
    s, prov = _sub_const(_NB_HYPHEN_NL, "", s, prov)
    s, prov = _sub_const(_NB_NL, " ", s, prov)
    s, prov = _sub_const(_NB_TAB, " ", s, prov)
    s, prov = _strip_prov(s, prov)
    # clean_text
    for _rid, pat, repl in CLEAN_RULES:
        s, prov = _sub_const(pat, repl, s, prov)
    s, prov = _sub_const(_WS, " ", s, prov)
    s, prov = _sub_group1(_SPACE_PUNCT, s, prov)
    s, prov = _strip_prov(s, prov)
    ref, _ = clean_text(normalise_block("\n".join(box_texts)))
    if s != ref or len(prov) != len(s):        # replay drifted — caller falls back
        return None, None
    return s, prov


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(vol: str, chapter: int = 1) -> Path:
    cfg = config.VIETNAMESE
    out_dir = config.OUT_DIR / vol
    manifest = read_json(out_dir / "split_manifest.json")
    vi_pages = manifest["vi_pages"]
    if not vi_pages:
        raise ValueError(f"no Vietnamese pages in manifest for {vol} — run step 1 first")

    pdf_path = config.DATA_DIR / f"{vol}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    page_box_texts: dict[int, list[str]] = {}   # page -> OCR line texts
    from .vi_ocr import VietnameseOCR
    dpi = config.SPLIT["render_dpi"]
    log.info("[%s] re-OCRing %d Vietnamese pages with Surya @ %d DPI ...",
             vol, len(vi_pages), dpi)
    save_dir = (out_dir / "pages_vi") if cfg.get("cache_images", True) else None
    engine = VietnameseOCR(cfg, dpi)
    vi_boxes: dict[int, list] = {}
    page_text = engine.ocr_pages(pdf_path, vi_pages, save_dir=save_dir,
                                 box_sink=vi_boxes)
    raw_pages = [page_text.get(p, "") for p in vi_pages]
    # Per-line detection boxes (coords in the cached pages_vi/*.png frame) for
    # the QA overlay in pipeline/draw_boxes.py. Sentence rows can't carry these
    # — a sentence is re-segmented across lines/pages — so persist them raw.
    box_rows = []
    for p in vi_pages:
        for b_idx, (bbox, txt) in enumerate(vi_boxes.get(p, []), start=1):
            box_rows.append({
                "id": make_id(chapter, p, b_idx),
                "chapter": chapter, "page": p, "box_idx": b_idx,
                "bbox": [int(round(c)) for c in bbox], "text": txt,
            })
        page_box_texts[p] = [t for _, t in vi_boxes.get(p, [])]
    write_jsonl(out_dir / "vi_boxes.jsonl", box_rows)

    fixer = SpellFixer(cfg)
    corrector = CorrectionMap(config.VI_CORRECTIONS_CSV)
    split = get_sentence_splitter()
    vocab = load_vi_vocab()

    # Concatenate the cleaned pages into ONE stream before sentence splitting so
    # a sentence that spans a page break (tail of page N + head of page N+1) is
    # segmented as one sentence instead of two fragments. Header / folio furniture
    # is stripped per page FIRST: a running head sitting between two pages would
    # otherwise fuse the tail of N and the head of N+1 into a single sentence.
    # Each sentence is attributed to the page its FIRST character falls on
    # (a cross-page sentence belongs to the page it starts on).
    from collections import defaultdict

    big = ""
    bounds: list[tuple[int, int]] = []        # (start_offset, page_no), in page order
    prov: list = []                           # parallel to big: (page, box_idx) | None per char
    SEP = " "                                 # space: a cross-page sentence joins cleanly
    for page_no, raw in zip(vi_pages, raw_pages):
        texts = page_box_texts.get(page_no)
        block, bprov = (None, None)
        if texts is not None:                 # track char -> box provenance
            block, bprov = block_with_prov(texts, page_no)
            if block is None:
                log.warning("[%s] page %d: provenance replay drifted — no box_ids",
                            vol, page_no)
        if block is None:                     # replay-drift fallback
            block, _ = clean_text(normalise_block(raw))
            bprov = [None] * len(block)
        bounds.append((len(big), page_no))
        big += block + SEP
        prov.extend(bprov)
        prov.append(None)                     # the SEP char

    def boxes_in(a: int, b: int) -> list[str]:
        """Ordered, de-duplicated vi_box ids touched by big[a:b]."""
        seen, ids = set(), []
        for key in prov[a:b]:
            if key is None or key in seen:
                continue
            seen.add(key)
            ids.append(make_id(chapter, key[0], key[1]))
        return ids

    def page_at(pos: int) -> int:
        pg = bounds[0][1]
        for off, p in bounds:
            if off > pos:
                break
            pg = p
        return pg

    rows = []
    cursor = 0
    sent_idx: dict[int, int] = defaultdict(int)
    for sent in split(big):
        start = big.find(sent, cursor)         # locate the sentence to map offset -> page
        if start < 0:
            start = cursor
        cursor = start + len(sent)
        page_no = page_at(start)
        fixed = corrector.apply(sent)          # domain fixes first
        fixed = fixer.fix_text(fixed)          # then generic spell fix
        fixed, _ = clean_text(fixed)                            # strip leftover noise
        if is_junk(fixed):                                      # row was only page furniture
            continue
        sent_idx[page_no] += 1
        n_tok, oov = oov_rate(fixed, vocab)   # QC signal for the step-4 review lane
        rows.append({
            "id": make_id(chapter, page_no, sent_idx[page_no]),
            "chapter": chapter,
            "page": page_no,
            "sent_idx": sent_idx[page_no],
            "text": fixed,
            "raw_text": sent,
            "n_tokens": n_tok,
            "oov_rate": round(oov, 3),
            "box_ids": boxes_in(start, cursor),   # vi_box ids this sentence covers
        })

    out_path = out_dir / "vi_sentences.jsonl"
    n = write_jsonl(out_path, rows)
    log.info("[%s] wrote %d Vietnamese sentences -> %s", vol, n, out_path.name)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default="vol1")
    ap.add_argument("--chapter", type=int, default=1)
    args = ap.parse_args()
    run(args.vol, args.chapter)


if __name__ == "__main__":
    main()

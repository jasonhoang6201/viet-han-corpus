"""Hán sentence segmentation — auto-punctuation of unpunctuated woodblock text.

Woodblock prints carry NO punctuation, so step 3 falls back to "one column == one
sentence". But a classical sentence flows across several columns (a VI sentence maps
to ~8 Hán columns), so column-as-sentence over-segments the Hán side ~3x and wrecks
the bge sentence alignment. This module re-segments the CLEAN (post-consensus) Hán
character stream into real sentences using an auto-punctuation model.

Split scope is the **whole chapter**: all pages' main columns are concatenated in
reading order into ONE character stream, punctuated, then split — so a sentence that
runs across a column OR page boundary stays whole (nothing is dropped). Because a
sentence can now span pages, ``box_ids`` are the GLOBAL box ``id`` strings (not
page-local indices); each sentence's ``page`` is where its first box sits. draw_boxes
resolves a box id straight to its (page, bbox). step4 / the aligner key on the
sentence id + text and are unaffected.

The heavy model I/O (a token-classification punctuator, e.g.
``raynardj/classical-chinese-punctuation-guwen-biaodian``) lives in notebook ②; this
module is dependency-free (stdlib only) so it stays testable. The notebook passes in
``labels_fn(text) -> list[str]`` returning, for each input character, the punctuation
mark predicted to follow it ("" for none) — it slides a fixed window with overlap over
the long stream internally, so no sentence is cut at a model-window boundary. We split
on sentence-final marks and map each sentence back to the columns it covers.

雙行 (interlinear-note) columns are NOT punctuated — their reading order is only a
geometric guess (``is_dbl``). Each 雙行 column is emitted as its own sentence, tagged
``is_dbl=True`` so the aligner can isolate / down-weight it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

# sentence-final marks the punctuator may emit (。！？；). ； (semicolon) is included:
# in classical prose it typically closes a full clause that aligns like a sentence.
SENT_MARKS = "。！？；"


def _split_stream(chars: list[str], marks: list[str], sent_marks: str) -> list[tuple[int, int]]:
    """Return [start, end) spans over `chars`, breaking after any sentence-final mark.

    `marks[k]` is the punctuation predicted to follow `chars[k]` ("" for none). A
    trailing run with no closing mark is emitted as a final sentence.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(chars)
    for k in range(n):
        if marks[k] and marks[k] in sent_marks:
            spans.append((start, k + 1))
            start = k + 1
    if start < n:
        spans.append((start, n))
    return spans


def resegment_boxes(han_boxes: list[dict],
                    labels_fn: Callable[[str], list[str]],
                    transliterate: Callable[[str, dict], str],
                    hanviet: dict,
                    make_id: Callable[[int, int, int], str],
                    sent_marks: str = SENT_MARKS) -> list[dict]:
    """Rebuild han_sentences rows from (corrected) han_boxes via auto-punctuation.

    Per chapter: concatenate every non-雙行 column's chars — across all pages, in
    (page, box_idx) reading order — into ONE stream, get per-char marks from
    ``labels_fn``, split on sentence-final marks. A sentence may span columns and
    pages; nothing is dropped. Each 雙行 column stays its own tagged sentence.

    Row schema matches step3.run(): {id, chapter, page, sent_idx, sinonom,
    am_han_viet, box_ids, is_dbl}. ``box_ids`` are GLOBAL box ``id`` strings; ``page``
    is the sentence's first box's page. Rows come out in reading order (main + 雙行
    interleaved by their first box), sent_idx numbered per page.
    """
    chapters: dict[int, list[dict]] = defaultdict(list)
    for b in han_boxes:
        chapters[b["chapter"]].append(b)

    def _emit(chapter: int, seg: str, boxes: list[dict], is_dbl: bool, pos: int) -> dict:
        # unique box ids in first-appearance (reading) order
        seen, ids = set(), []
        for b in boxes:
            if b["id"] not in seen:
                seen.add(b["id"]); ids.append(b["id"])
        return {"_pos": pos, "chapter": chapter, "page": boxes[0]["page"],
                "sinonom": seg, "am_han_viet": transliterate(seg, hanviet),
                "box_ids": ids, "is_dbl": is_dbl}

    out: list[dict] = []
    for chapter in sorted(chapters):
        cols = sorted(chapters[chapter], key=lambda b: (b["page"], b["box_idx"]))
        pos_of = {b["id"]: i for i, b in enumerate(cols)}   # global reading-order index

        main = [b for b in cols if not b.get("is_dbl")]
        chars: list[str] = []
        owner: list[dict] = []
        for b in main:
            for ch in b["sinonom"]:
                chars.append(ch)
                owner.append(b)

        rows: list[dict] = []
        if chars:
            marks = labels_fn("".join(chars))
            if len(marks) != len(chars):                    # defensive: labels_fn must align
                marks = (list(marks) + [""] * len(chars))[:len(chars)]
            for (a, e) in _split_stream(chars, marks, sent_marks):
                seg = "".join(chars[a:e])
                if seg:
                    rows.append(_emit(chapter, seg, owner[a:e], False, pos_of[owner[a]["id"]]))

        for b in cols:
            if b.get("is_dbl") and b["sinonom"]:
                rows.append(_emit(chapter, b["sinonom"], [b], True, pos_of[b["id"]]))

        rows.sort(key=lambda r: r.pop("_pos"))              # reading order (main + 雙行)
        per_page: dict[int, int] = defaultdict(int)
        for r in rows:
            per_page[r["page"]] += 1
            r["sent_idx"] = per_page[r["page"]]
            r["id"] = make_id(chapter, r["page"], r["sent_idx"])
            out.append(r)
    return out

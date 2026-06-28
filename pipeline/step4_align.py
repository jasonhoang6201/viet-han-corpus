"""Step 4 — Align the Hán and Vietnamese sides, validate chars, export Excel.

Inputs (from steps 2 & 3):
  out/<vol>/vi_sentences.jsonl
  out/<vol>/han_sentences.jsonl
  out/<vol>/han_boxes.jsonl

Outputs:
  out/<vol>/alignment.jsonl          sentence alignment (m-n) with similarity
  out/<vol>/char_validation.jsonl    per-char S1∩S2 colour status
  out/<vol>/<PREFIX>_alignment.xlsx  spec workbook (box-level + sentence sheet)

Two things happen here, following SinoNom_OCR_TransliterationAlignment.pdf.
Sentence alignment itself runs in notebook ② (bge-m3 dense+sparse); this module
only does the char validation, review queues and Excel export.

1. Character validation (§I, the S1∩S2 rule): for each (sn, qn) pair
        sn ∈ S2(qn)            -> BLACK (OCR correct)
        else G = S1(sn)∩S2(qn) -> len 1: GREEN, >1: GREEN(best in S1 order),
                                   0: RED (OCR failure)

2. Excel export (§II): ID | Image box | SinoNom char | Âm Hán Việt |
   Nghĩa thuần Việt, one row per box, ID = DSG_fff.ccc.ppp.ss.

Run:  python -m pipeline.step4_align --vol vol1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .common import get_logger, load_vi_vocab, read_jsonl, vi_tokens, write_jsonl

log = get_logger("step4")


# --------------------------------------------------------------------------- #
# Dictionaries for char validation
# --------------------------------------------------------------------------- #
def _parse_dic(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, _, rest = line.partition(":")
        out[k.strip()] = [t for t in rest.split() if t.strip()]
    return out


class CharValidator:
    """Implements the S1∩S2 colouring rule from the course guide."""

    BLACK, GREEN, RED = "BLACK", "GREEN", "RED"

    def __init__(self):
        self.s1 = _parse_dic(config.SINONOM_SIMILAR_DIC)              # sn -> [similar]
        self.s2 = {k: set(v) for k, v in _parse_dic(config.QUOCNGU_SINONOM_DIC).items()}

    def validate(self, sn: str, qn: str) -> dict:
        s2 = self.s2.get(qn, set())
        if sn in s2:
            return {"sn": sn, "qn": qn, "status": self.BLACK, "corrected": sn}
        s1 = self.s1.get(sn, [sn])
        inter = [c for c in s1 if c in s2]                            # keep S1 order
        if len(inter) >= 1:
            return {"sn": sn, "qn": qn, "status": self.GREEN, "corrected": inter[0]}
        return {"sn": sn, "qn": qn, "status": self.RED, "corrected": None}


def build_review(boxes: list[dict], char_rows: list[dict], cfg: dict) -> list[dict]:
    """Build the human-review queue: chars/boxes a person should check.

    Two independent failure modes (a confidence threshold alone misses the
    second — OCR is often *confidently* wrong):
      * RED   — char's reading is not consistent with any valid SinoNom char
                (out-of-dictionary or mis-recognised). Flagged per character.
      * low_conf — the whole box was read with low OCR confidence (blurry/
                damaged plate). Flagged per box.
    Each row has empty `fix_type` / `correct` fields for the reviewer to fill in
    (fix_type: dict_gap | ocr_wrong | drop).
    """
    from collections import defaultdict

    box_by_id = {b["id"]: b for b in boxes}
    rows: list[dict] = []
    pos_in_box: dict[str, int] = defaultdict(int)
    for r in char_rows:
        pos = pos_in_box[r["id"]]
        pos_in_box[r["id"]] += 1
        if r["status"] != "RED":
            continue
        b = box_by_id.get(r["id"], {})
        text = b.get("sinonom", "")
        context = f"{text[:pos]}【{r['sn']}】{text[pos + 1:]}"   # mark the char in its column
        rows.append({
            "level": "char", "reason": "RED",
            "id": r["id"], "page": r["page"], "box_idx": r["box_idx"],
            "char_pos": pos, "sn": r["sn"], "qn": r["qn"],
            "conf": b.get("conf"), "bbox": b.get("bbox"),
            "context": context,
            "fix_type": "", "correct": "",   # reviewer fills: dict_gap|ocr_wrong|drop  +  reading or char
        })

    red_ids = {r["id"] for r in char_rows if r["status"] == "RED"}
    for b in boxes:
        conf = b.get("conf")
        if conf is None or conf >= cfg["conf_threshold"]:
            continue
        rows.append({
            "level": "box", "reason": "low_conf",
            "id": b["id"], "page": b["page"], "box_idx": b["box_idx"],
            "char_pos": None, "sn": b.get("sinonom", ""), "qn": b.get("am_han_viet", ""),
            "conf": conf, "bbox": b.get("bbox"),
            "context": b.get("sinonom", ""),
            "fix_type": "", "correct": "",
            "also_red": b["id"] in red_ids,
        })
    return rows


def build_vi_review(vie: list[dict], vocab: set[str], cfg: dict) -> list[dict]:
    """Flag Vietnamese sentences with a high out-of-vocabulary token rate.

    Complements the Hán lanes: `low_conf` catches blurry boxes, RED catches
    content errors, and `high_oov` catches the VI OCR reading non-words on the VI
    side (where there is no model confidence). Uses the stored `oov_rate` if the
    sentence row carries one (full step-2 run), else computes it on the fly so a
    step-4-only rerun still works on existing vi_sentences.jsonl.
    """
    from .common import oov_rate, oov_tokens

    thr = cfg.get("oov_threshold", 0.5)
    mint = cfg.get("min_oov_tokens", 4)
    rows = []
    for v in vie:
        n, oov = v.get("n_tokens"), v.get("oov_rate")
        if oov is None or n is None:
            n, oov = oov_rate(v["text"], vocab)
        if n < mint or oov < thr:
            continue
        rows.append({
            "level": "vi_sentence", "reason": "high_oov",
            "id": v["id"], "page": v["page"],
            "oov_rate": round(oov, 3), "n_tokens": n,
            "oov_tokens": oov_tokens(v["text"], vocab),   # which words to check
            "text": v["text"],
            "fix_type": "", "correct": "",   # reviewer: ocr_wrong | drop | ok
        })
    return rows


def build_oov_vocab(vie: list[dict], vocab: set[str]) -> list[dict]:
    """Aggregate EVERY out-of-vocab token across ALL VI sentences (no threshold).

    The review lane shows garbled *sentences*; this shows the distinct OOV
    *tokens* with frequency so the dict can be grown: high count = a real word
    missing from Viet74K (reign title, Hán-Việt term) worth whitelisting; count
    1 + odd shape = OCR garble. Sorted by count desc, then token."""
    from collections import defaultdict
    from .common import vi_tokens

    count: dict[str, int] = defaultdict(int)
    example: dict[str, str] = {}
    for v in vie:
        for t in set(vi_tokens(v["text"])):     # set: one sentence counts once
            if vocab and t not in vocab:
                count[t] += 1
                example.setdefault(t, v["id"])
    rows = [{"token": t, "count": c, "example_id": example[t]}
            for t, c in count.items()]
    rows.sort(key=lambda r: (-r["count"], r["token"]))
    return rows


def validate_boxes(boxes: list[dict], validator: CharValidator) -> list[dict]:
    rows = []
    for b in boxes:
        sylls = b["am_han_viet"].split()
        chars = list(b["sinonom"])
        for ch, qn in zip(chars, sylls):
            r = validator.validate(ch, qn)
            rows.append({"id": b["id"], "page": b["page"], "box_idx": b["box_idx"], **r})
    return rows


# --------------------------------------------------------------------------- #
# Sentence-alignment helpers (used by notebook ②'s bge-m3 aligner)
# --------------------------------------------------------------------------- #
def lexical_sim(am_han_viet: str, vie_text: str) -> float:
    """Jaccard token overlap between the Hán-Việt reading and the VI translation.

    A lexical second opinion on each pair, independent of the neural model.
    Vietnamese keeps a large Sino-Vietnamese vocabulary, so a correctly aligned
    pair shares tokens ("trung thần", "bản triều") even when the neural model
    underrates classical Hán. Cheap and immune to văn-ngôn weakness; blind only
    when the translator uses a native-Việt paraphrase ("mở nước" for 開國 "khai
    quốc"). Paired with the dense score the two cover each other's blind spots —
    see ALIGN.suspect_* in config.
    """
    a = set(vi_tokens(am_han_viet))
    b = set(vi_tokens(vie_text))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_vi_sentences(vie: list[dict], cfg: dict) -> tuple[list[dict], int]:
    """Drop front-matter / OCR-garbage VI sentences before alignment.

    The VI side carries title pages, library stamps and publisher lines whose
    embedded-OCR text is noise (e.g. '0UẾC-SỬ-0UẦN tuều NGUYÊN'). Aligning the
    real Hán content against them produces near-zero-similarity junk pairs and
    drags the whole DP off by an offset. Keep a sentence only when enough of its
    alphabetic tokens are real Vietnamese words.
    """
    import re

    vocab = load_vi_vocab()
    if not vocab:
        return vie, 0
    kept, dropped = [], 0
    for v in vie:
        toks = [t for t in re.findall(r"[^\W\d_]+", v["text"], flags=re.UNICODE)]
        if len(toks) < cfg["min_vi_tokens"]:
            dropped += 1
            continue
        hit = sum(1 for t in toks if t.lower() in vocab)
        if hit / len(toks) < cfg["min_vi_invocab_ratio"]:
            dropped += 1
            continue
        kept.append(v)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Excel export
# --------------------------------------------------------------------------- #
def export_excel(vol: str, boxes: list[dict], pairs: list[dict],
                 char_rows: list[dict], path: Path) -> None:
    import pandas as pd

    # Map each Hán sentence id -> aligned Vietnamese text (Nghĩa thuần Việt).
    sent_meaning = {}
    for p in pairs:
        for hid in p["han_ids"]:
            sent_meaning[hid] = p["vietnamese"]

    # Box-level sheet (spec layout). A box inherits the meaning of the sentence
    # it belongs to; boxes map 1-1 to sentences in the unpunctuated case.
    box_records = []
    for b in boxes:
        box_records.append({
            "ID": b["id"],
            "Image box": str(b["bbox"]),
            "SinoNom char": b["sinonom"],
            "Âm Hán Việt": b["am_han_viet"],
            "Nghĩa thuần Việt": sent_meaning.get(b["id"], ""),
        })
    df_box = pd.DataFrame(box_records)
    df_sent = pd.DataFrame([{
        "Hán IDs": ", ".join(p["han_ids"]),
        "Việt IDs": ", ".join(p["vie_ids"]),
        "SinoNom": p["sinonom"],
        "Âm Hán Việt": p["am_han_viet"],
        "Nghĩa thuần Việt": p["vietnamese"],
        "Similarity": p["similarity"],
        "Lexical": p.get("lexical", ""),
        "Nghi lệch": "x" if p.get("suspect") else "",
    } for p in pairs])
    df_char = pd.DataFrame(char_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df_box.to_excel(xl, sheet_name="boxes", index=False)
        df_sent.to_excel(xl, sheet_name="sentence_alignment", index=False)
        df_char.to_excel(xl, sheet_name="char_validation", index=False)
    log.info("[%s] wrote %s (%d boxes, %d pairs)", vol, path.name, len(df_box), len(df_sent))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(vol: str) -> None:
    out_dir = config.OUT_DIR / vol
    boxes = list(read_jsonl(out_dir / "han_boxes.jsonl"))
    han_sents = list(read_jsonl(out_dir / "han_sentences.jsonl"))
    vie_sents = list(read_jsonl(out_dir / "vi_sentences.jsonl"))

    log.info("[%s] %d boxes, %d Hán sents, %d Việt sents", vol, len(boxes), len(han_sents), len(vie_sents))

    validator = CharValidator()
    char_rows = validate_boxes(boxes, validator)
    write_jsonl(out_dir / "char_validation.jsonl", char_rows)
    n_black = sum(1 for r in char_rows if r["status"] == "BLACK")
    n_green = sum(1 for r in char_rows if r["status"] == "GREEN")
    n_red = sum(1 for r in char_rows if r["status"] == "RED")
    log.info("[%s] char validation: BLACK=%d GREEN=%d RED=%d", vol, n_black, n_green, n_red)

    vi_vocab = load_vi_vocab()
    review_rows = build_review(boxes, char_rows, config.REVIEW)
    review_rows += build_vi_review(vie_sents, vi_vocab, config.REVIEW)
    write_jsonl(out_dir / "review.jsonl", review_rows)

    oov_vocab = build_oov_vocab(vie_sents, vi_vocab)
    write_jsonl(out_dir / "oov_vocab.jsonl", oov_vocab)
    log.info("[%s] OOV vocab: %d distinct OOV tokens -> oov_vocab.jsonl (grow the dict from here)",
             vol, len(oov_vocab))
    n_low = sum(1 for r in review_rows if r["reason"] == "low_conf")
    n_oov = sum(1 for r in review_rows if r["reason"] == "high_oov")
    n_red = sum(1 for r in review_rows if r["reason"] == "RED")
    log.info("[%s] review queue: %d items (%d RED chars + %d low-conf boxes + %d high-oov VI) -> review.jsonl",
             vol, len(review_rows), n_red, n_low, n_oov)

    # Sentence alignment runs in notebook ② (bge-m3); this module only emits the
    # char-validation, review queues and the box/char Excel sheets.
    pairs: list[dict] = []

    s = config.ID_SCHEMA
    prefix = f"{s['domain']}{s['subdomain']}{s['genre']}_{s['file_id']}"
    export_excel(vol, boxes, pairs, char_rows, out_dir / f"{prefix}_alignment.xlsx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default="vol1")
    args = ap.parse_args()
    run(args.vol)


if __name__ == "__main__":
    main()

"""Step 4 — Align the Hán and Vietnamese sides, export the spec Excel (P3).

P3 is alignment-only. The VI review queue is built in step 2 (P1) and the Hán
review queue in step 3 (P2, after consensus) — see pipeline/reviews.py. This
module consumes their outputs.

Inputs (from steps 2 & 3):
  out/<vol>/vi_sentences.jsonl
  out/<vol>/han_sentences.jsonl
  out/<vol>/han_boxes.jsonl

Outputs:
  out/<vol>/<PREFIX>_alignment.xlsx  spec workbook (box-level + sentence sheet)

Sentence alignment itself runs in notebook ③ (bge-m3 dense+sparse); this module
emits the box Excel sheet (§II): ID | Image box | SinoNom char |
Âm Hán Việt | Nghĩa thuần Việt, one row per box, ID = DSG_fff.ccc.ppp.ss.

Run:  python -m pipeline.step4_align --vol vol1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .common import get_logger, load_vi_vocab, read_jsonl, vi_tokens

log = get_logger("step4")


# --------------------------------------------------------------------------- #
# The VI/Hán review queues moved to pipeline/reviews.py: the VI lane runs in
# step 2 (P1) and the Hán lane in step 3 (P2, after the Qwen consensus). This
# module is alignment + Excel only.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Sentence-alignment helpers (used by notebook ③'s bge-m3 aligner)
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
                 path: Path) -> None:
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
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df_box.to_excel(xl, sheet_name="boxes", index=False)
        df_sent.to_excel(xl, sheet_name="sentence_alignment", index=False)
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

    # Review queues run in their own stages (VI lane in P1 step 2, Hán lane in P2
    # step 3 after consensus). P3 is alignment-only.
    #
    # Sentence alignment runs in notebook ③ (bge-m3); this module only emits the
    # box Excel sheet from the alignment pairs.
    pairs: list[dict] = []

    s = config.ID_SCHEMA
    prefix = f"{s['domain']}{s['subdomain']}{s['genre']}_{s['file_id']}"
    export_excel(vol, boxes, pairs, out_dir / f"{prefix}_alignment.xlsx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default="vol1")
    args = ap.parse_args()
    run(args.vol)


if __name__ == "__main__":
    main()

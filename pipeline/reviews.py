"""Human-review queues — flag the lines/chars a person should check.

Split by stage so each half of the pipeline writes its own queue (P3 is now
alignment-only):

  * VI lane  (P1, step 2): `vi_review.jsonl` + `oov_vocab.jsonl`
      high_oov — a Vietnamese sentence whose out-of-vocabulary token rate is high
      (the VI OCR likely read non-words). The only quality signal on the VI side,
      where there is no model confidence.
  * Hán lane (P2, step 3, AFTER consensus): `han_review.jsonl`
      low_conf — the box was read with low consensus confidence. OCR correctness
      is judged by the 3-engine consensus + Qwen arbiter (see han_consensus), so
      the review queue only surfaces the low-confidence boxes a human should
      re-check. Every review row carries empty `fix_type` / `correct` fields for
      the reviewer to fill in.
"""
from __future__ import annotations

from collections import defaultdict

from .common import oov_rate, oov_tokens, vi_tokens


# --------------------------------------------------------------------------- #
# Vietnamese lane (P1)
# --------------------------------------------------------------------------- #
def build_vi_review(vie: list[dict], vocab: set[str], cfg: dict) -> list[dict]:
    """Flag Vietnamese sentences with a high out-of-vocabulary token rate.

    Uses the stored `oov_rate` if the sentence row carries one (full step-2 run),
    else computes it on the fly so a rerun still works on existing sentences.
    """
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


# --------------------------------------------------------------------------- #
# Hán lane (P2) — low-confidence box review (OCR correctness comes from the
# consensus + Qwen arbiter, not a dictionary rule-check)
# --------------------------------------------------------------------------- #
def build_review(boxes: list[dict], cfg: dict) -> list[dict]:
    """Build the Hán human-review queue: low-confidence boxes.

    The real OCR-correctness signal is the 3-engine consensus + Qwen arbiter
    (see han_consensus): a box read with low consensus confidence is what a human
    should re-check. The old dictionary rule-check (S1∩S2 char validation) was
    dropped — its reading was derived from the OCR char itself, so the check was
    near-circular and only ever flagged chars missing from the reading dict.
    """
    rows: list[dict] = []
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
        })
    return rows

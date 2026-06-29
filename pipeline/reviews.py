"""Human-review queues — flag the lines/chars a person should check.

Split by stage so each half of the pipeline writes its own queue (P3 is now
alignment-only):

  * VI lane  (P1, step 2): `vi_review.jsonl` + `oov_vocab.jsonl`
      high_oov — a Vietnamese sentence whose out-of-vocabulary token rate is high
      (the VI OCR likely read non-words). The only quality signal on the VI side,
      where there is no model confidence.
  * Hán lane (P2, step 3, AFTER consensus): `char_validation.jsonl` + `han_review.jsonl`
      RED      — a char's reading is consistent with no valid SinoNom char.
      low_conf — the whole box was read with low OCR confidence.

A confidence threshold alone misses RED: OCR is often *confidently* wrong, so the
dictionary cross-check (S1∩S2) is a separate lane. Every review row carries empty
`fix_type` / `correct` fields for the reviewer to fill in.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import config
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
# Hán lane (P2) — char validation (S1∩S2 rule from the course guide)
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
    """Implements the S1∩S2 colouring rule from the course guide.

        sn ∈ S2(qn)            -> BLACK (OCR correct)
        else G = S1(sn)∩S2(qn) -> len>=1: GREEN(best in S1 order); 0: RED (failure)
    """

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


def validate_boxes(boxes: list[dict], validator: CharValidator) -> list[dict]:
    rows = []
    for b in boxes:
        sylls = b["am_han_viet"].split()
        chars = list(b["sinonom"])
        for ch, qn in zip(chars, sylls):
            r = validator.validate(ch, qn)
            rows.append({"id": b["id"], "page": b["page"], "box_idx": b["box_idx"], **r})
    return rows


def build_review(boxes: list[dict], char_rows: list[dict], cfg: dict) -> list[dict]:
    """Build the Hán human-review queue: RED chars + low-confidence boxes."""
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

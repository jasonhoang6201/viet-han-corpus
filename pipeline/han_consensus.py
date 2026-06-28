"""Hán OCR consensus — model-independent voting + re-transliteration.

The heavy model I/O (specialist PaddleOCR rec, Qwen2.5-VL) lives in notebook ②;
this module holds the *pure* logic so it is testable and reusable:

  * fold()            — collapse traditional/simplified so engines compare fairly
  * map_to_base()     — align another engine's reading onto base-char positions
  * vote_column()     — per-character arbiter vote (qwen_arbiter | majority)
  * build_consensus() — vote every column -> consensus records + review queue
  * apply_consensus() — rewrite han boxes/sentences with the corrected chars and
                        re-derive âm Hán-Việt, so the bge aligner sees the fix
  * consensus_metrics()

Method follows notebooks/Minh_Menh_4_Han_Consensus_Pipeline.ipynb (the validated
ad-hoc run): base = PaddleOCR (step 3), spec = MinhDS fine-tuned PP-OCRv5 rec,
qwen = Qwen2.5-VL arbiter. In `qwen_arbiter` mode base==qwen keeps base, a
disagreement trusts qwen and is flagged for review; `spec` is only recorded as a
backing signal (it is currently weak, see consensus_metrics in [[mmcy-readme-config-drift]]).

The module is dependency-free (stdlib + optional `opencc`); the notebook loads
the dictionary and passes it in, so importing this never pulls in paddle/torch.
"""
from __future__ import annotations

import collections
import difflib

# --------------------------------------------------------------------------- #
# Traditional/simplified fold (so 安 vs 安, 機 vs 机 compare equal across engines)
# --------------------------------------------------------------------------- #
try:
    from opencc import OpenCC

    _t2s = OpenCC("t2s")

    def fold(s: str) -> str:
        return _t2s.convert(s or "")
except Exception:  # opencc not installed -> compare raw (still correct, less lenient)
    def fold(s: str) -> str:
        return s or ""


# --------------------------------------------------------------------------- #
# âm Hán-Việt (self-contained copy of step3.transliterate to avoid importing
# paddle/torch through the step3 module at notebook-import time)
# --------------------------------------------------------------------------- #
def _is_cjk(c: str) -> bool:
    o = ord(c)
    return 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF


def transliterate(chars: str, hanviet: dict[str, str]) -> str:
    return " ".join(hanviet.get(c, "?") for c in chars if _is_cjk(c))


# --------------------------------------------------------------------------- #
# Per-character alignment + vote
# --------------------------------------------------------------------------- #
def map_to_base(base: str, other: str) -> dict[int, str]:
    """Map each base-char position to the corresponding char in `other`.

    Uses a folded SequenceMatcher so equal/replace spans line up even when the
    two engines disagree on traditional vs. simplified forms. Insertions in
    `other` are dropped (we vote on the base backbone, never lengthen a column).
    """
    bf, of = fold(base), fold(other)
    sm = difflib.SequenceMatcher(None, bf, of)
    mp: dict[int, str] = {}
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(min(a2 - a1, b2 - b1)):
                mp[a1 + k] = other[b1 + k]
    return mp


def vote_column(base: str, spec: str, qwen: str,
                vote_mode: str = "qwen_arbiter") -> tuple[str, list[dict]]:
    """Vote one column's characters. Returns (consensus_text, review_positions).

    qwen_arbiter — base==qwen keeps base; qwen absent keeps base; a disagreement
                   trusts qwen and records the position for human review. `spec`
                   is informational only (spec_backs_qwen).
    majority     — keep a char only when >=2 engines agree on it, else keep base
                   and flag.
    """
    smap = map_to_base(base, spec) if spec else {}
    qmap = map_to_base(base, qwen) if qwen else {}
    chars: list[str] = []
    review: list[dict] = []
    for i, bc in enumerate(base):
        sc = smap.get(i)
        qc = qmap.get(i) if qwen else None
        if vote_mode == "qwen_arbiter":
            if not qc:                              # qwen skipped this column -> keep base
                chars.append(bc)
            elif fold(qc) == fold(bc):              # base & qwen agree -> trust base
                chars.append(bc)
            else:                                   # disagree -> trust qwen, flag review
                chars.append(qc)
                review.append({"pos": i, "base": bc, "spec": sc, "qwen": qc,
                               "spec_backs_qwen": bool(sc and fold(sc) == fold(qc))})
        else:                                       # majority
            cand = {"base": bc, "spec": sc, "qwen": qc}
            votes = collections.Counter(fold(c) for c in cand.values() if c)
            top, cnt = votes.most_common(1)[0]
            if cnt >= 2:
                chars.append(next(c for c in cand.values() if c and fold(c) == top))
            else:
                chars.append(bc)
                review.append({"pos": i, "base": bc, "spec": sc, "qwen": qc})
    return "".join(chars), review


def build_consensus(records: list[dict],
                    vote_mode: str = "qwen_arbiter") -> tuple[list[dict], list[dict]]:
    """records: [{id, page, conf, base, spec, qwen}]. Returns (consensus, review).

    consensus row: {id, page, conf, consensus, base, spec, qwen, n_review}
    review row   : {id, page, base, spec, qwen, positions:[...]} (only flagged cols)
    """
    consensus, review = [], []
    for r in records:
        base = r.get("base") or ""
        spec = r.get("spec") or ""
        qwen = r.get("qwen") or ""
        text, col_review = vote_column(base, spec, qwen, vote_mode)
        consensus.append({"id": r["id"], "page": r.get("page"), "conf": r.get("conf"),
                          "consensus": text, "base": base, "spec": spec, "qwen": qwen,
                          "n_review": len(col_review)})
        if col_review:
            review.append({"id": r["id"], "page": r.get("page"),
                           "base": base, "spec": spec, "qwen": qwen, "positions": col_review})
    return consensus, review


# --------------------------------------------------------------------------- #
# Apply corrections back onto the box / sentence rows the aligner reads
# --------------------------------------------------------------------------- #
def apply_consensus(rows: list[dict], corrected: dict[str, str],
                    hanviet: dict[str, str]) -> list[dict]:
    """Rewrite han box/sentence rows with the consensus characters.

    For every row whose id is in `corrected`, replace `sinonom` with the corrected
    text and re-derive `am_han_viet` from the corrected characters (so both the
    bge dense signal — built on sinonom — and the sparse signal — built on âm
    Hán-Việt — see the fix). Rows absent from `corrected` are returned unchanged.

    Assumes one column == one sentence (unpunctuated woodblock print), where the
    box id equals the sentence id; this holds for the Minh Mệnh corpus.
    """
    out = []
    for row in rows:
        new = {k: v for k, v in row.items() if not k.startswith("_")}  # drop _spec/_qwen scratch
        fixed = corrected.get(row["id"])
        if fixed is not None and fixed != row.get("sinonom"):
            new["sinonom"] = fixed
            new["am_han_viet"] = transliterate(fixed, hanviet)
        out.append(new)
    return out


def consensus_metrics(consensus: list[dict], review: list[dict], vol: str,
                      vote_mode: str, vlm_mode: str, qwen_cols_run: int) -> dict:
    tot = sum(len(r["base"]) for r in consensus)
    nrev = sum(r["n_review"] for r in consensus)
    nfix = sum(1 for r in consensus if r["base"] != r["consensus"])
    return {
        "vol": vol, "cols": len(consensus), "chars": tot,
        "review_cols": len(review), "review_chars": nrev, "corrected_chars": nfix,
        "auto_settle_char_pct": round(100 - nrev / max(1, tot) * 100, 2),
        "vote_mode": vote_mode, "vlm_mode": vlm_mode, "qwen_cols_run": qwen_cols_run,
    }

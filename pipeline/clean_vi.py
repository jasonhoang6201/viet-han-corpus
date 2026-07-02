"""Clean OCR noise from the Vietnamese sentence layer.

Page furniture (running headers, folio markers) and OCR math hallucinations
(\\mathbf{x}, \\bullet, ... from the math model misreading ornamental dots)
bleed into the recognised text. This module strips them with the rules
documented in docs/vi_noise_patterns.txt — keep the two in sync.

Two entry points:
  * clean_text(text)            -> cleaned str  (used inline by step2)
  * run(vol)                    -> post-process out/<vol>/vi_sentences.jsonl
                                   in place, writing vi_clean_report.json.

raw_text is never touched, so cleaning is reversible. Rows whose text becomes
empty / letterless after cleaning are dropped (and listed in the report).

Run standalone:  python -m pipeline.clean_vi --vol vol1
"""
from __future__ import annotations

import argparse
import re

from . import config
from .common import get_logger, load_vi_vocab, oov_rate, read_jsonl, write_jsonl
from .reviews import build_oov_vocab, build_vi_review

log = get_logger("clean_vi")

# Ordered (id, pattern, replacement). Order matters: kill LaTeX + folio first so
# the header words sit contiguous, then strip the header, then the leftover
# leading page number. See docs/vi_noise_patterns.txt for the rationale.
CLEAN_RULES: list[tuple[str, re.Pattern, str]] = [
    # 4. LaTeX / math hallucinations (decorative section dots, bold caps)
    ("LATEX_BRACED", re.compile(r"\\[a-zA-Z]+\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"), " "),
    ("LATEX_PAREN",  re.compile(r"\\[a-zA-Z]+\s*\([^()]*\)"), " "),
    ("SUBSCRIPT",    re.compile(r"[A-Za-z]?_\{[^{}]*\}"), " "),
    ("LATEX_STAR",   re.compile(r"\\\*"), " "),
    ("LATEX_BARE",   re.compile(r"\\[a-zA-Z]+"), " "),
    # 5. folio / leaf markers
    ("FOLIO_BRKT",   re.compile(r"\[\s*\d{1,3}\s*[-.\s]?\s*[abAB.]?\s*\]"), " "),
    ("FOLIO_PAREN",  re.compile(r"\(\s*\d{1,3}\s*[-.\s]?\s*[abAB]\s*\)"), " "),
    # 1./3. running header + front-matter (UPPERCASE anchors -> prose is safe)
    ("HEADER_TITLE", re.compile(
        r"(?:\b\d{1,3}\s+)?(?:[A-Za-zÀ-ỹ.]+\s+){1,3}?CH[IÍ]NH[\s-]*Y[ÉẾEÊÉ]U\b"), " "),
    ("FRONTMATTER",  re.compile(r"\bQUY[ÊẾE]N\s+TH[ỦU]\b"), " "),
    ("PUBLISHER",    re.compile(
        r"\b(?:T[UỦ]\s*S[AÁ]CH\s+C[OỔ]\s+V[ĂÄA]N|UY\s+BAN\s+DICH\s+THUAT"
        r"|PH[ỦU]\s+QU[ỐO]C\s+V[ỤU]\s+KHANH)\b"), " "),
    # 2. stray leading page number left behind
    ("LEAD_PAGENUM", re.compile(r"^\s*\d{1,3}\s+(?=[A-Za-zÀ-ỹ«])"), ""),
    # 6. decorative separators
    ("DECOR",        re.compile(r"(?<=\s)[*•](?=\s|$)"), " "),
]

# dangling "[NN." opener produced by a bad sentence split — whole row is junk
_FOLIO_OPEN = re.compile(r"^\s*[\[(]?\s*\d{1,3}\s*[-.\s]?\s*[abAB.]?\s*$")
_HAS_LETTER = re.compile(r"[A-Za-zÀ-ỹ]")
_WS = re.compile(r"\s{2,}")
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?»])")


def clean_text(text: str) -> tuple[str, list[str]]:
    """Return (cleaned_text, [rule_ids that fired])."""
    fired: list[str] = []
    for rid, pat, repl in CLEAN_RULES:
        new = pat.sub(repl, text)
        if new != text:
            fired.append(rid)
            text = new
    text = _WS.sub(" ", text)
    text = _SPACE_PUNCT.sub(r"\1", text)
    return text.strip(), fired


def is_junk(text: str) -> bool:
    """True if the cleaned text carries no real content."""
    return not _HAS_LETTER.search(text) or bool(_FOLIO_OPEN.match(text))


def run(vol: str, path=None) -> dict:
    from pathlib import Path
    if path is not None:
        path = Path(path)
        out_dir = path.parent
    else:
        out_dir = config.OUT_DIR / vol
        path = out_dir / "vi_sentences.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    vocab = load_vi_vocab()
    rows = list(read_jsonl(path))
    kept: list[dict] = []
    changed, dropped = [], []
    rule_hits: dict[str, int] = {}

    for row in rows:
        before = row.get("text", "")
        after, fired = clean_text(before)
        for rid in fired:
            rule_hits[rid] = rule_hits.get(rid, 0) + 1
        if is_junk(after):
            dropped.append({"id": row.get("id"), "text": before})
            continue
        if after != before:
            changed.append({"id": row.get("id"), "before": before, "after": after})
            row["text"] = after
            row["n_tokens"], oov = oov_rate(after, vocab)   # keep QC signal in sync
            row["oov_rate"] = round(oov, 3)
        kept.append(row)

    write_jsonl(path, kept)

    # VI review queue — built here (the last stage to touch vi_sentences.jsonl) so
    # it reflects the FINAL corpus, and here rather than step 4 so the Vietnamese
    # side can be hand-reviewed right after P1, before any Hán OCR / alignment runs.
    write_jsonl(out_dir / "vi_review.jsonl", build_vi_review(kept, vocab, config.REVIEW))
    write_jsonl(out_dir / "oov_vocab.jsonl", build_oov_vocab(kept, vocab))
    report = {
        "vol": vol,
        "rows_in": len(rows),
        "rows_out": len(kept),
        "rows_changed": len(changed),
        "rows_dropped": len(dropped),
        "rule_hits": dict(sorted(rule_hits.items(), key=lambda kv: -kv[1])),
        "dropped": dropped,
        "changed_sample": changed[:50],
    }
    rep_path = out_dir / "vi_clean_report.json"
    rep_path.write_text(
        __import__("json").dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log.info("[%s] cleaned: %d in -> %d out (%d changed, %d dropped) -> %s",
             vol, len(rows), len(kept), len(changed), len(dropped), path.name)
    log.info("[%s] rule hits: %s", vol, report["rule_hits"])
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean OCR noise from vi_sentences.jsonl")
    ap.add_argument("--vol", default="vol1")
    ap.add_argument("--path", default=None,
                    help="explicit path to a vi_sentences.jsonl (overrides --vol/OUT_DIR)")
    args = ap.parse_args()
    run(args.vol, path=args.path)


if __name__ == "__main__":
    main()

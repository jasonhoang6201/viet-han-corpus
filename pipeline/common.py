"""Shared utilities: logging, JSONL IO, ID-schema formatting, text helpers."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

from . import config


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def progress(iterable, desc: str, total: int | None = None, log=None):
    """Wrap an iterable with a visible progress bar so every pipeline phase
    reports how far along it is.

    Uses tqdm (renders inline in Colab) when available; otherwise falls back to
    periodic INFO logs every ~10% so a headless run still shows movement. `total`
    is inferred from `len(iterable)` when omitted.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="it")
    except Exception:                           # pragma: no cover - tqdm optional
        pass

    def _gen():
        step = max(1, (total or 0) // 10) if total else 0
        for i, item in enumerate(iterable, 1):
            if log is not None and step and (i % step == 0 or i == total):
                log.info("%s %d/%s", desc, i, total)
            yield item
    return _gen()


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
def paddle_device() -> str:
    """Return "gpu" if paddle is CUDA-compiled with a visible GPU, else "cpu".

    Step 3 / VI detection use this to pick GPU (server models, no oneDNN) over
    CPU (mobile models + oneDNN). Any import/probe failure -> "cpu".
    """
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# JSON / JSONL IO
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# ID schema:  DSG_fff.ccc.ppp.ss
# --------------------------------------------------------------------------- #
def make_id(chapter: int, page: int, sentence: int, schema: dict | None = None) -> str:
    s = schema or config.ID_SCHEMA
    prefix = f"{s['domain']}{s['subdomain']}{s['genre']}_{s['file_id']}"
    return f"{prefix}.{chapter:03d}.{page:03d}.{sentence:02d}"


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
# Matches a Vietnamese "word" token (latin letters + Vietnamese diacritics).
VI_TOKEN_RE = re.compile(
    r"[a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+",
    re.IGNORECASE,
)

# CJK ideograph ranges (BMP + common extensions present in SinoNom corpora).
_CJK_RANGES = (
    (0x3400, 0x4DBF),    # Ext A
    (0x4E00, 0x9FFF),    # URO
    (0xF900, 0xFAFF),    # Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Ext B
    (0x2A700, 0x2EBEF),  # Ext C–F
    (0x2F800, 0x2FA1F),  # Compatibility Supplement
)


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def cjk_chars(text: str) -> list[str]:
    return [c for c in text if is_cjk(c)]


def vi_tokens(text: str) -> list[str]:
    return VI_TOKEN_RE.findall(text.lower())


# Vietnamese-specific accented letters (base vowels + tone marks + đ).
VI_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
)


def vi_diacritic_ratio(text: str) -> tuple[int, float]:
    """Return (token_count, fraction of tokens containing a Vietnamese diacritic)."""
    toks = vi_tokens(text)
    if not toks:
        return 0, 0.0
    accented = sum(1 for t in toks if any(c in VI_DIACRITICS for c in t))
    return len(toks), accented / len(toks)


def load_vi_vocab(single_syllable: bool = False) -> set[str]:
    """Lowercased Vietnamese wordlist from `config.VIET_WORDLIST`.

    Shared by step 2 (spell-fix + OOV) and step 4 (VI filter + OOV report) so the
    list is loaded one way everywhere. `single_syllable=True` keeps only one-token
    words (no space/hyphen) — what the edit-distance SpellFixer needs; the default
    full set is right for OOV lookups (single-token queries never match multiword
    entries anyway)."""
    vocab: set[str] = set()
    if not config.VIET_WORDLIST.exists():
        return vocab
    for line in config.VIET_WORDLIST.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if not w:
            continue
        if single_syllable and (" " in w or "-" in w):
            continue
        vocab.add(w)
    return vocab


def oov_rate(text: str, vocab: set[str]) -> tuple[int, float]:
    """Return (token_count, fraction of tokens NOT found in `vocab`).

    Tokens are lowercased Vietnamese word tokens (`vi_tokens`). High OOV means the
    OCR likely produced non-words. An empty vocab or no tokens yields rate 0.0 —
    we never flag when we cannot judge. Soft signal: proper nouns and Hán-Việt
    terms absent from the wordlist inflate it."""
    toks = vi_tokens(text)
    if not toks or not vocab:
        return len(toks), 0.0
    oov = sum(1 for t in toks if t not in vocab)
    return len(toks), oov / len(toks)


def oov_tokens(text: str, vocab: set[str]) -> list[str]:
    """The distinct lowercased tokens of `text` absent from `vocab`, in order.

    Lets the review queue name *which* words to check (the VI analogue of the
    Hán RED chars), not just the aggregate `oov_rate`. Empty vocab -> []."""
    if not vocab:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for t in vi_tokens(text):
        if t not in vocab and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

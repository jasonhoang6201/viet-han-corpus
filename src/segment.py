"""B3 · Tách câu cho cả hai phía.

Việt: dùng underthesea.sent_tokenize nếu có, fallback regex theo dấu câu.
Hán: văn ngôn không có dấu câu -> tách theo dấu cú đậu cổ (。！？；：、) nếu
OCR ra được, fallback gộp theo cột/độ dài. Mỗi 'câu' Hán thực dụng = một bbox
cột (đã sắp thứ tự) khi thiếu dấu câu.
"""
from __future__ import annotations
import re

# ---- Việt ------------------------------------------------------------------
_VI_FALLBACK = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ])")


def split_viet(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        from underthesea import sent_tokenize
        sents = sent_tokenize(text)
    except Exception:
        sents = _VI_FALLBACK.split(text.replace("\n", " "))
    return [s.strip() for s in sents if s.strip()]


# ---- Hán -------------------------------------------------------------------
_HAN_PUNCT = "。！？；：、，"
_HAN_SPLIT = re.compile(f"(?<=[{_HAN_PUNCT}])")
_HAN_CHAR = re.compile(r"[㐀-鿿豈-﫿]")


def split_han(text: str) -> list[str]:
    """Tách câu Hán theo dấu cú đậu; bỏ dấu, chỉ giữ chữ Hán."""
    parts = _HAN_SPLIT.split(text)
    out = []
    for p in parts:
        chars = "".join(_HAN_CHAR.findall(p))
        if chars:
            out.append(chars)
    return out


def han_chars(text: str) -> str:
    return "".join(_HAN_CHAR.findall(text))

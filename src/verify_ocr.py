"""B3-Hán · Kiểm OCR đúng/sai theo thuật toán file SinoNom (char alignment).

So 1 chữ Hán OCR (sn) với 1 âm Quốc ngữ tương ứng (qn):
  S1 = chữ giống hình với sn (SinoNom_Similar.dic)
  S2 = chữ khả dĩ đọc thành qn (QuocNgu_SinoNom.dic)
  - sn ∈ S2            -> 'black'  (OCR đúng)
  - |S1∩S2| == 1       -> 'green'  (sửa thành chữ đó)
  - |S1∩S2| > 1        -> 'green'  (chọn giống sn nhất qua Levenshtein)
  - |S1∩S2| == 0       -> 'red'    (OCR sai, không xác định)
Thiếu SinoNom_Similar.dic -> chế độ rút gọn: chỉ phân biệt black/red theo S2.
"""
from __future__ import annotations
from dataclasses import dataclass
from . import dicts


@dataclass
class CharCheck:
    sn: str          # chữ Hán OCR
    qn: str          # âm Quốc ngữ kỳ vọng
    color: str       # black | green | red
    fixed: str       # chữ sau sửa (== sn nếu không sửa)


def _lev(a: str, b: str) -> int:
    try:
        import Levenshtein
        return Levenshtein.distance(a, b)
    except Exception:
        # fallback DP nhỏ
        if a == b:
            return 0
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            for j in range(1, n + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                             prev[j - 1] + (a[i - 1] != b[j - 1]))
            prev = cur
        return prev[n]


def check_char(sn: str, qn: str) -> CharCheck:
    S2 = dicts.quocngu_sinonom().get(qn, set())
    if sn in S2:
        return CharCheck(sn, qn, "black", sn)
    S1 = dicts.sinonom_similar().get(sn, set())
    if not S1:                       # chế độ rút gọn (thiếu Similar.dic)
        return CharCheck(sn, qn, "red", sn)
    inter = S1 & S2
    if len(inter) == 1:
        return CharCheck(sn, qn, "green", next(iter(inter)))
    if len(inter) > 1:
        best = min(inter, key=lambda c: _lev(c, sn))
        return CharCheck(sn, qn, "green", best)
    return CharCheck(sn, qn, "red", sn)


def check_box(han: str, readings: list[str]) -> list[CharCheck]:
    """Kiểm cả một bbox: chuỗi chữ Hán + list âm QN tương ứng từng chữ."""
    out = []
    for i, sn in enumerate(han):
        qn = readings[i] if i < len(readings) else ""
        out.append(check_char(sn, qn))
    return out

"""Từ điển dùng cho dịch âm (B3) và kiểm OCR (verify).

Có sẵn (đã fetch):
  dicts/phienam.txt   — char=âm_Hán_Việt (11k mục), nguồn cộng đồng dịch CV.

Suy ra:
  hanviet[char]  -> âm Hán Việt (cho cột 'Âm Hán Việt' của Excel).
  qn2sn[âm]      -> {các chữ Hán đọc thành âm đó}  ≈ QuocNgu_SinoNom.dic (S2).

Tùy chọn (lấy từ thầy nếu có) — đặt vào dicts/:
  QuocNgu_SinoNom.dic   (âm QN -> tập chữ Hán Nôm)  : ưu tiên hơn bản suy ra.
  SinoNom_Similar.dic   (chữ -> tập chữ giống hình, S1) : để tinh chỉnh sửa OCR.
Nếu thiếu SinoNom_Similar.dic, bước verify chạy chế độ rút gọn (chỉ S2).
"""
from __future__ import annotations
import os
import functools
from . import config


def _read_lines(name: str) -> list[str]:
    p = os.path.join(config.DICTS_DIR, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


@functools.lru_cache(maxsize=1)
def hanviet() -> dict[str, str]:
    """char -> âm Hán Việt (1 âm chính)."""
    d = {}
    for line in _read_lines("phienam.txt"):
        if "=" in line:
            c, r = line.split("=", 1)
            if c and r and c not in d:
                d[c.strip()] = r.strip().lower()
    return d


def reading_of(text: str) -> str:
    """Dịch âm cả chuỗi Hán -> 'nam quốc sơn hà' (chữ thiếu -> '?')."""
    hv = hanviet()
    return " ".join(hv.get(c, "?") for c in text)


@functools.lru_cache(maxsize=1)
def quocngu_sinonom() -> dict[str, set[str]]:
    """S2: âm QN -> tập chữ Hán. Ưu tiên file thầy, fallback đảo từ phienam."""
    d: dict[str, set[str]] = {}
    lines = _read_lines("QuocNgu_SinoNom.dic")
    if lines:
        for line in lines:
            # định dạng giả định: "âm\tchữ1 chữ2 ..."  hoặc "âm:chữ1,chữ2"
            sep = "\t" if "\t" in line else (":" if ":" in line else None)
            if not sep:
                continue
            k, v = line.split(sep, 1)
            chars = [c for c in v.replace(",", " ").split() if c]
            d.setdefault(k.strip().lower(), set()).update(chars)
        return d
    # fallback: đảo phienam
    for c, r in hanviet().items():
        d.setdefault(r, set()).add(c)
    return d


@functools.lru_cache(maxsize=1)
def sinonom_similar() -> dict[str, set[str]]:
    """S1: chữ -> tập chữ giống hình. Rỗng nếu không có file thầy."""
    d: dict[str, set[str]] = {}
    for line in _read_lines("SinoNom_Similar.dic"):
        sep = "\t" if "\t" in line else (":" if ":" in line else None)
        if not sep:
            continue
        k, v = line.split(sep, 1)
        chars = [c for c in v.replace(",", " ").split() if c]
        d.setdefault(k.strip(), set()).update(chars)
    return d


def status() -> dict[str, int | bool]:
    return {
        "hanviet_chars": len(hanviet()),
        "qn2sn_keys": len(quocngu_sinonom()),
        "has_real_QuocNgu_SinoNom": bool(_read_lines("QuocNgu_SinoNom.dic")),
        "has_SinoNom_Similar": len(sinonom_similar()) > 0,
    }


if __name__ == "__main__":
    print(status())
    print("test:", reading_of("南國山河"))

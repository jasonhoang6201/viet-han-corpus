"""B1 · Tách trang Hán / trang Việt trong mỗi PDF.

Lớp text của trang Hán là rác Tesseract (chữ Latin lộn xộn) -> đếm số từ
không đáng tin. Thay vào đó đếm số ÂM TIẾT TIẾNG VIỆT HỢP LỆ dài >=3 ký tự
(đối chiếu syllables.txt). Trang dịch có hàng trăm, trang Hán có <10.

Kết quả mỗi tập: phần đầu = Việt (dịch), 1 block lớn cuối = Hán (gốc).
"""
from __future__ import annotations
import re
import functools
from . import config


@functools.lru_cache(maxsize=1)
def _syllables() -> frozenset[str]:
    with open(config.SYLLABLES_TXT, encoding="utf-8") as f:
        return frozenset(l.strip().lower() for l in f if l.strip())


_TOK = re.compile(r"[a-zA-Zà-ỹÀ-ỸđĐ]+")


def good3_count(text: str) -> int:
    """Số âm tiết tiếng Việt hợp lệ (>=3 ký tự) trong text."""
    syl = _syllables()
    return sum(1 for w in (m.group().lower() for m in _TOK.finditer(text))
               if len(w) >= 3 and w in syl)


def classify_pages(pdf_path: str) -> list[str]:
    """Trả về nhãn 'V'/'H' cho từng trang."""
    import fitz  # lazy: PyMuPDF
    doc = fitz.open(pdf_path)
    labels = []
    for i in range(len(doc)):
        g = good3_count(doc[i].get_text())
        labels.append("V" if g >= config.VIET_PAGE_MIN_GOOD3 else "H")
    doc.close()
    return labels


def split_point(labels: list[str]) -> int:
    """Chỉ số trang bắt đầu block Hán lớn nhất (cuối sách).

    Các H-run ngắn (<=2) nằm trong phần Việt = trang trắng/ngăn chương -> bỏ qua.
    """
    best_start, best_len, i, n = None, 0, 0, len(labels)
    while i < n:
        if labels[i] == "H":
            j = i
            while j < n and labels[j] == "H":
                j += 1
            run = j - i
            if run > best_len:
                best_len, best_start = run, i
            i = j
        else:
            i += 1
    return best_start if best_start is not None else n


def split_volume(pdf_path: str) -> dict:
    """{'viet': [page_idx...], 'han': [page_idx...], 'split_at': int}."""
    labels = classify_pages(pdf_path)
    sp = split_point(labels)
    viet = [i for i in range(sp) if labels[i] == "V"]   # bỏ trang trắng đầu sách
    han = list(range(sp, len(labels)))
    return {"viet": viet, "han": han, "split_at": sp, "labels": labels}


if __name__ == "__main__":
    for p in config.list_books():
        r = split_volume(p)
        print(f"{config.nfc(p.split('/')[-1]):40s} "
              f"split@{r['split_at']:>4}  viet={len(r['viet']):>4}  han={len(r['han']):>4}")

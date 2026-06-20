"""Central config for the Minh Mệnh Chính Yếu Hán–Việt parallel-corpus pipeline.

Đề tài 39 (HVB): ngữ liệu song song Hán–Việt, lịch sử VN.
Mỗi PDF = phần dịch Quốc ngữ (đầu sách) + phần Hán gốc (cuối sách, chỉ là ảnh).
"""
from __future__ import annotations
import os
import unicodedata

# ---- Paths -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS_DIR = os.path.join(ROOT, "thien_menh_chinh_yeu_books")
SYLLABLES_TXT = os.path.join(ROOT, "syllables.txt")
DICTS_DIR = os.path.join(ROOT, "dicts")
OUT_DIR = os.path.join(ROOT, "out")
DEBUG_DIR = os.path.join(ROOT, "debug_img")

# ---- ID scheme (file SinoNom): DSG_fff.ccc.ppp.ss --------------------------
# DSG_fff do thầy cấp. Tạm dùng placeholder cho pilot.
ID_PREFIX = "HBA_039"   # H=History, B=Base, A=genre; 039 = số đề tài

# ---- Page classification (B1) ----------------------------------------------
# Đếm số "âm tiết tiếng Việt hợp lệ, dài >=3 ký tự" trong lớp text mỗi trang.
# Trang dịch (Việt) có hàng trăm; trang Hán (lớp text là rác Tesseract) có <10.
VIET_PAGE_MIN_GOOD3 = 30

# ---- OCR backend (B2-Hán) --------------------------------------------------
# "paddle" | "mock"  (Gemini/API cố ý không dùng — chạy trên Colab Pro GPU)
OCR_BACKEND = os.environ.get("OCR_BACKEND", "paddle")
OCR_DPI = 300                       # render trang PDF -> ảnh
OCR_LANG = "chinese_cht"            # PaddleOCR: phồn thể, gần với Hán Nôm khắc gỗ
COLUMN_X_TOLERANCE = 25             # ngưỡng gom bbox vào cùng một cột (px @300dpi sẽ scale)

# ---- Alignment (B4) --------------------------------------------------------
LABSE_MODEL = "sentence-transformers/LaBSE"
ALIGN_MIN_SIM = 0.50               # ngưỡng cosine tối thiểu để chấp nhận cặp câu
ALIGN_MAX_MERGE = 3                # cho phép gộp tối đa m,n câu (m-n alignment)

# ---- Metadata cho XML ------------------------------------------------------
META = {
    "title": "Minh Mệnh Chính Yếu",
    "author": "Quốc Sử Quán triều Nguyễn",
    "era": "Nhà Nguyễn",
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def list_books() -> list[str]:
    """Trả về đường dẫn 6 PDF, sắp theo số tập."""
    import re
    fs = [f for f in os.listdir(BOOKS_DIR) if f.lower().endswith(".pdf")]
    def vol(f):
        m = re.search(r"vol(\d+)", nfc(f))
        return int(m.group(1)) if m else 0
    return [os.path.join(BOOKS_DIR, f) for f in sorted(fs, key=vol)]

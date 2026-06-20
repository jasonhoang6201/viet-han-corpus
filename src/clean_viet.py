"""B2-V · Lấy & làm sạch bản dịch Quốc ngữ từ lớp text PDF.

Lớp OCR sẵn còn nhiều lỗi (vd 'EHÍNH VẾU'). Ở đây làm sạch BẢO THỦ:
chuẩn hóa Unicode, gỡ header/số trang chạy, nối từ bị ngắt dòng, gom khoảng
trắng. Sửa lỗi chính tả sâu cần LLM (đề bài cho phép) — để hook riêng.
"""
from __future__ import annotations
import re
from . import config

# Header chạy thường gặp ở mép trang (đã viết hoa), và số trang đứng một mình.
_RUNNING_HEAD = re.compile(
    r"^\s*\d{0,4}\s*(MINH[\s.\-]*MỆNH|MINH[\s.\-]*MỆNH[\s.\-]*CHÍNH[\s.\-]*YẾU|"
    r"QUYỀN|QUYỂN|QUYEN)[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_PAGE_NUM = re.compile(r"^\s*[\dIVXLC]{1,5}\s*$", re.MULTILINE)
_FOOTNOTE_MARK = re.compile(r"\(\s*\d{1,2}\s*\)|\(\s*[a-zđ]\s*\)|†")
_DEHYPHEN = re.compile(r"([a-zà-ỹ])-\n([a-zà-ỹ])")   # từ bị ngắt cuối dòng
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    t = config.nfc(text)
    t = _DEHYPHEN.sub(r"\1\2", t)
    t = _RUNNING_HEAD.sub("", t)
    t = _PAGE_NUM.sub("", t)
    t = _FOOTNOTE_MARK.sub("", t)
    # nối dòng trong cùng đoạn: \n đơn không sau dấu câu -> khoảng trắng
    t = re.sub(r"(?<![.!?:;])\n(?=[a-zà-ỹ0-9])", " ", t)
    t = _MULTISPACE.sub(" ", t)
    t = _MULTINL.sub("\n\n", t)
    return t.strip()


def extract_viet(pdf_path: str, viet_pages: list[int]) -> dict[int, str]:
    """{page_idx: cleaned_text} cho các trang Việt."""
    import fitz
    doc = fitz.open(pdf_path)
    out = {}
    for i in viet_pages:
        out[i] = clean_text(doc[i].get_text())
    doc.close()
    return out

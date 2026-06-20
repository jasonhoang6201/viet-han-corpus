"""B2-Hán · OCR trang Hán (ảnh) -> chữ Hán + bbox. Backend cắm rời.

Chạy trên Colab Pro GPU, KHÔNG dùng API. Mặc định backend 'paddle' (PaddleOCR,
miễn phí, trả bbox sẵn). Có thể đổi qua biến môi trường OCR_BACKEND.

Mỗi backend nhận đường dẫn ảnh -> list[Box] (text = chữ Hán trong box, đã là
cấp cột/dòng). Sau đó dùng sort_bbox.reading_order để sắp.

Backend 'mock': đọc file <ảnh>.gt.txt (mỗi dòng 1 cột) để chạy thử pipeline
không cần GPU.
"""
from __future__ import annotations
import os
from . import config
from .sort_bbox import Box


# ---- render PDF -> ảnh ------------------------------------------------------
def page_to_image(pdf_path: str, page_idx: int, out_png: str, dpi: int | None = None) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    pix = doc[page_idx].get_pixmap(dpi=dpi or config.OCR_DPI)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pix.save(out_png)
    doc.close()
    return out_png


# ---- backends ---------------------------------------------------------------
_PADDLE = None


def _paddle_ocr(image_path: str) -> list[Box]:
    """PaddleOCR. Cài trên Colab: pip install paddleocr paddlepaddle-gpu."""
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR
        _PADDLE = PaddleOCR(use_angle_cls=True, lang=config.OCR_LANG, show_log=False)
    result = _PADDLE.ocr(image_path, cls=True)
    boxes: list[Box] = []
    for line in (result[0] or []):
        quad, (text, conf) = line[0], line[1]
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        boxes.append(Box(min(xs), min(ys), max(xs), max(ys),
                         text=text, extra={"conf": float(conf)}))
    return boxes


def _mock_ocr(image_path: str) -> list[Box]:
    gt = os.path.splitext(image_path)[0] + ".gt.txt"
    boxes: list[Box] = []
    if not os.path.exists(gt):
        return boxes
    # giả lập cột dọc: cột i đặt bên phải, x giảm dần
    cols = [l.strip() for l in open(gt, encoding="utf-8") if l.strip()]
    W = 100 * len(cols)
    for i, col in enumerate(cols):
        x = W - (i + 1) * 100
        boxes.append(Box(x, 0, x + 80, 40 * len(col), text=col))
    return boxes


_BACKENDS = {"paddle": _paddle_ocr, "mock": _mock_ocr}


def ocr_image(image_path: str, backend: str | None = None) -> list[Box]:
    b = backend or config.OCR_BACKEND
    fn = _BACKENDS.get(b)
    if fn is None:
        raise ValueError(f"OCR backend không hỗ trợ: {b}")
    return fn(image_path)

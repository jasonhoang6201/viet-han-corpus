# Đề tài 39 — Ngữ liệu song song Hán–Việt (Minh Mệnh Chính Yếu)

Pipeline xây ngữ liệu song song Hán–Việt từ 6 PDF *Minh Mệnh Chính Yếu*.
Yêu cầu B.1 (HVB): OCR Hán → dóng hàng dịch âm (ký tự) & dịch nghĩa (câu) →
xuất **XML** (thẻ `<C>`/`<V>`, Sentence_ID 14 ký tự) + **Excel** (mức ký tự).

## Cấu trúc dữ liệu (đã khảo sát thực tế)
Mỗi PDF = **phần dịch Quốc ngữ (đầu sách)** + **một block Hán gốc (cuối sách,
chỉ là ảnh, lớp text là rác Tesseract)**. Điểm cắt mỗi tập:

| Tập | Trang | Việt (đầu) | Hán (cuối) |
|----|----|----|----|
| 1 | 638 | 0–223 | 224–637 |
| 2 | 416 | 0–145 | 146–415 |
| 3 | 738 | 0–309 | 310–737 |
| 4 | 516 | 0–213 | 214–515 |
| 5 | 566 | 0–217 | 218–565 |
| 6 | 736 | 0–397 | 398–735 |

Phân loại trang dùng `syllables.txt`: đếm số âm tiết tiếng Việt hợp lệ dài ≥3
ký tự (trang dịch hàng trăm, trang Hán <10) — bền hơn đếm từ thô vì rác
Tesseract trên trang Hán giả dạng âm tiết ngắn.

## Pipeline (2 nhánh)
```
B1 split  ─┬─ Việt: clean → tách câu ─────────────┐
           └─ Hán: render ảnh → OCR(+bbox) →       ├─ B4 align câu (anchor+LaBSE, m-n)
              sắp cột RTL → dịch âm → kiểm OCR ─────┘   → B5 ID + XML/Excel → B6 eval
```

Module trong `src/`: `split_pages` `clean_viet` `segment` `ocr_han`
`sort_bbox` `dicts` `verify_ocr` `align` `export` `evaluate`.

## Chạy
```bash
python -m run_pipeline split  --vol 1     # tách trang  (máy thường)
python -m run_pipeline viet   --vol 1     # clean + tách câu Việt (máy thường)
python -m run_pipeline han    --vol 1     # OCR Hán + bbox + dịch âm  (CẦN GPU)
python -m run_pipeline align  --vol 1     # dóng hàng câu LaBSE       (CẦN GPU)
python -m run_pipeline export --vol 1     # XML + Excel (máy thường)
python -m run_pipeline all    --vol 1 --limit 5   # full, giới hạn 5 trang Hán để thử
```
Kết quả vào `out/vol<N>/`. Backend OCR đổi bằng `OCR_BACKEND=mock|paddle`.

## Trên Colab Pro (GPU, không dùng API)
```python
!pip install -q pymupdf openpyxl python-Levenshtein underthesea \
    paddleocr paddlepaddle-gpu sentence-transformers
!python -m run_pipeline all --vol 1
```

## Phụ thuộc từ điển (`dicts/`)
- `phienam.txt` — **đã có** (11.411 char→âm Hán Việt). Dùng cho cột *Âm Hán Việt*
  và suy ra QuocNgu_SinoNom (S2) bằng cách đảo map.
- `QuocNgu_SinoNom.dic`, `SinoNom_Similar.dic` — **xin từ thầy** rồi bỏ vào
  `dicts/`. Có thì `verify_ocr` chạy đủ (đen/xanh/đỏ); thiếu `SinoNom_Similar.dic`
  thì chạy rút gọn (chỉ đen/đỏ theo S2).

## Hạn chế đã biết (cần xử lý để tăng F1)
1. **Lớp OCR tiếng Việt rất bẩn** (`Minhmệnh`, `Triệu.tri`). Nên OCR lại trang
   Việt bằng engine khác (VietOCR/PaddleOCR-vi) — thay `clean_viet.extract_viet`
   bằng bộ OCR. Hiện làm sạch bảo thủ + fallback regex tách câu.
2. **Recognition Hán trên bản khắc gỗ** của PaddleOCR có thể yếu → kiểm bằng
   thuật toán SinoNom (`verify_ocr`) và cần `SinoNom_Similar.dic` để sửa.
3. Chưa có **golden** để đo P/R/F1 (`evaluate.prf1` sẵn sàng khi có nhãn).
4. `phienam.txt` có chữ giản thể & 1 âm/chữ; chữ đa âm Hán Việt chỉ lấy âm chính.

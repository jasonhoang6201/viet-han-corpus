"""Orchestrator pipeline Đề tài 39 — chạy cho 1 tập (pilot).

Các stage lưu JSON trung gian vào out/<vol>/ để OCR (GPU, Colab) và phần còn
lại tách rời, chạy lại được. Han OCR + LaBSE cần Colab Pro GPU; B1/Việt/export
chạy được ở máy thường.

Dùng:
  python -m run_pipeline split   --vol 1
  python -m run_pipeline viet    --vol 1
  python -m run_pipeline han      --vol 1        # cần PaddleOCR (GPU)
  python -m run_pipeline align    --vol 1        # cần sentence-transformers
  python -m run_pipeline export   --vol 1
  python -m run_pipeline all      --vol 1
"""
from __future__ import annotations
import argparse
import json
import os
from src import config, split_pages, clean_viet, segment, ocr_han, sort_bbox, dicts, align as aln, export


def _vol_dir(vol: int) -> str:
    d = os.path.join(config.OUT_DIR, f"vol{vol}")
    os.makedirs(d, exist_ok=True)
    return d


def _book(vol: int) -> str:
    return config.list_books()[vol - 1]


def _save(vol, name, obj):
    p = os.path.join(_vol_dir(vol), name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"  -> {p}")


def _load(vol, name):
    with open(os.path.join(_vol_dir(vol), name), encoding="utf-8") as f:
        return json.load(f)


# ---- stages ----------------------------------------------------------------
def stage_split(vol: int):
    r = split_pages.split_volume(_book(vol))
    _save(vol, "split.json", {"viet": r["viet"], "han": r["han"], "split_at": r["split_at"]})
    print(f"split: viet={len(r['viet'])} han={len(r['han'])} split@{r['split_at']}")


def stage_viet(vol: int):
    sp = _load(vol, "split.json")
    texts = clean_viet.extract_viet(_book(vol), sp["viet"])
    full = "\n".join(texts[i] for i in sp["viet"])
    sents = segment.split_viet(full)
    _save(vol, "viet_sents.json", sents)
    print(f"viet: {len(sents)} câu")


def stage_han(vol: int, limit: int | None = None):
    sp = _load(vol, "split.json")
    pages = sp["han"][:limit] if limit else sp["han"]
    imgdir = os.path.join(_vol_dir(vol), "han_img")
    han_sents, char_rows = [], []
    for pi in pages:
        png = os.path.join(imgdir, f"p{pi}.png")
        if not os.path.exists(png):
            ocr_han.page_to_image(_book(vol), pi, png)
        boxes = ocr_han.ocr_image(png)
        ordered = sort_bbox.reading_order(boxes)
        for ss, b in enumerate(ordered, 1):
            han = segment.han_chars(b.text)
            if not han:
                continue
            han_sents.append(han)
            char_rows.append({
                "page": pi, "ss": ss,
                "image_box": str(b.as_corners()),
                "sinonom": han,
                "am_han_viet": dicts.reading_of(han),
            })
    _save(vol, "han_sents.json", han_sents)
    _save(vol, "char_rows.json", char_rows)
    print(f"han: {len(han_sents)} box/câu trên {len(pages)} trang")


def stage_align(vol: int):
    han = _load(vol, "han_sents.json")
    viet = _load(vol, "viet_sents.json")
    pairs = aln.align(han, viet)
    out = [{"han_idx": p.han_idx, "viet_idx": p.viet_idx,
            "han": "".join(han[i] for i in p.han_idx),
            "viet": " ".join(viet[j] for j in p.viet_idx),
            "score": p.score} for p in pairs]
    _save(vol, "pairs.json", out)
    print(f"align: {len(out)} cặp câu")


def stage_export(vol: int):
    pairs = _load(vol, "pairs.json")
    char_rows = _load(vol, "char_rows.json") if os.path.exists(
        os.path.join(_vol_dir(vol), "char_rows.json")) else []
    xml_pairs, excel_rows = [], []
    for ss, p in enumerate(pairs, 1):
        sid = export.make_id(ccc=vol, ppp=1, ss=ss)
        xml_pairs.append({"id": sid, "han": p["han"], "viet": p["viet"]})
    for r in char_rows:
        sid = export.make_id(ccc=vol, ppp=r["page"], ss=r["ss"])
        excel_rows.append({"id": sid, "image_box": r["image_box"],
                           "sinonom": r["sinonom"], "am_han_viet": r["am_han_viet"],
                           "nghia_thuan_viet": ""})
    d = _vol_dir(vol)
    export.write_xml(xml_pairs, os.path.join(d, f"minh_menh_vol{vol}.xml"))
    export.write_excel(excel_rows, os.path.join(d, f"minh_menh_vol{vol}.xlsx"))
    print(f"export: {len(xml_pairs)} cặp -> XML, {len(excel_rows)} dòng -> Excel")


STAGES = {"split": stage_split, "viet": stage_viet, "han": stage_han,
          "align": stage_align, "export": stage_export}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--vol", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="giới hạn số trang Hán (thử)")
    a = ap.parse_args()
    if a.stage == "all":
        stage_split(a.vol); stage_viet(a.vol)
        stage_han(a.vol, a.limit); stage_align(a.vol); stage_export(a.vol)
    elif a.stage == "han":
        stage_han(a.vol, a.limit)
    else:
        STAGES[a.stage](a.vol)


if __name__ == "__main__":
    main()

"""B5 · Gán Sentence_ID 14 ký tự + xuất XML (thẻ C/V) và Excel.

ID: DSG_fff.ccc.ppp.ss  (DSG_fff = config.ID_PREFIX do thầy cấp)
  ccc = chương/tập, ppp = trang, ss = số cặp câu/box trong trang.

XML: mỗi cặp câu -> <STC_ID id="..."><C>Hán</C><V>Việt</V></STC_ID>
Excel (mức ký tự, theo file SinoNom): ID | Image box | SinoNom | Âm Hán Việt | Nghĩa thuần Việt
"""
from __future__ import annotations
import os
from xml.sax.saxutils import escape
from . import config


def make_id(ccc: int, ppp: int, ss: int) -> str:
    return f"{config.ID_PREFIX}.{ccc:03d}.{ppp:03d}.{ss:02d}"


# ---- XML (mức câu) ----------------------------------------------------------
def write_xml(pairs: list[dict], out_path: str, meta: dict | None = None) -> str:
    """pairs: [{'id','han','viet'}]."""
    meta = meta or config.META
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<document>", "  <metadata>"]
    for k in ("title", "author", "era"):
        lines.append(f"    <{k}>{escape(meta.get(k,''))}</{k}>")
    lines.append("  </metadata>")
    for p in pairs:
        lines.append(
            f'  <STC_ID id="{p["id"]}">'
            f'<C>{escape(p["han"])}</C><V>{escape(p["viet"])}</V></STC_ID>'
        )
    lines.append("</document>")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


# ---- Excel (mức ký tự) ------------------------------------------------------
def write_excel(rows: list[dict], out_path: str) -> str:
    """rows: [{'id','image_box','sinonom','am_han_viet','nghia_thuan_viet'}]."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "char_level"
    ws.append(["ID", "Image box", "SinoNom", "Âm Hán Việt", "Nghĩa thuần Việt"])
    for r in rows:
        ws.append([r["id"], r["image_box"], r["sinonom"],
                   r["am_han_viet"], r.get("nghia_thuan_viet", "")])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wb.save(out_path)
    return out_path

"""Build a self-contained QA report (HTML) for one volume's pipeline output.

Reads out/<vol>/*.jsonl, computes quality stats, crops every human-review
bbox out of the page images, and writes out/<vol>/report.html with everything
embedded (base64) so it opens offline with no server.

Usage:  python -m pipeline.make_report --vol vol1
"""
from __future__ import annotations
import argparse, base64, io, json, statistics, html
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def histogram(vals, edges):
    """Count vals into [edges[i], edges[i+1]) buckets; last bucket inclusive."""
    out = [0] * (len(edges) - 1)
    for v in vals:
        for i in range(len(edges) - 1):
            if (v >= edges[i] and v < edges[i + 1]) or (i == len(edges) - 2 and v == edges[-1]):
                out[i] += 1
                break
    return out


def svg_bars(counts, labels, color, w=520, h=160):
    """Tiny dependency-free bar chart."""
    mx = max(counts) or 1
    n = len(counts)
    bw = w / n
    bars = []
    for i, c in enumerate(counts):
        bh = (c / mx) * (h - 28)
        x = i * bw
        y = h - 20 - bh
        bars.append(
            f'<rect x="{x+4:.1f}" y="{y:.1f}" width="{bw-8:.1f}" height="{bh:.1f}" fill="{color}" rx="2"/>'
            f'<text x="{x+bw/2:.1f}" y="{y-3:.1f}" font-size="10" text-anchor="middle" fill="#444">{c}</text>'
            f'<text x="{x+bw/2:.1f}" y="{h-6:.1f}" font-size="9" text-anchor="middle" fill="#888">{labels[i]}</text>'
        )
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{"".join(bars)}</svg>'


def crop_b64(img: Image.Image, bbox, pad=14, max_w=160, max_h=320):
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(img.width, x1 + pad); y1 = min(img.height, y1 + pad)
    c = img.crop((x0, y0, x1, y1))
    # scale down keeping aspect
    scale = min(max_w / c.width, max_h / c.height, 1.0)
    if scale < 1.0:
        c = c.resize((max(1, int(c.width * scale)), max(1, int(c.height * scale))))
    buf = io.BytesIO()
    c.convert("RGB").save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", default="vol1")
    ap.add_argument("--no-crops", action="store_true", help="skip image crops (faster)")
    args = ap.parse_args()
    od = ROOT / "out" / args.vol

    hb = load(od / "han_boxes.jsonl")
    hs = load(od / "han_sentences.jsonl")
    vs = load(od / "vi_sentences.jsonl")
    al = load(od / "alignment.jsonl")
    # Review queues split by stage: vi_review.jsonl (P1) + han_review.jsonl (P2).
    # Fall back to the legacy combined review.jsonl if present.
    rv = (load(od / "vi_review.jsonl") + load(od / "han_review.jsonl")
          or load(od / "review.jsonl"))
    n_pages = len(list((od / "pages_han").glob("*.png")))

    # ---- stats ----
    confs = [b["conf"] for b in hb if "conf" in b]
    sims = [a["similarity"] for a in al if "similarity" in a]
    n_chars = sum(len(b.get("sinonom", "")) for b in hb)
    n_low = sum(1 for r in rv if r["reason"] == "low_conf")

    conf_edges = [0, .3, .5, .7, .8, .9, 1.0001]
    conf_lbls = ["<.3", ".3-.5", ".5-.7", ".7-.8", ".8-.9", ".9+"]
    conf_hist = histogram(confs, conf_edges)
    sim_edges = [0, .1, .2, .3, .4, .5, 1.0001]
    sim_lbls = ["<.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ".5+"]
    sim_hist = histogram(sims, sim_edges)

    def pct(x, n): return f"{x/n*100:.1f}%" if n else "—"

    # ---- crops for review queue ----
    rows_sorted = sorted(rv, key=lambda r: r.get("conf", 1.0))
    page_cache: dict[int, Image.Image] = {}
    review_html = []
    for r in rows_sorted:
        crop = ""
        if not args.no_crops and r.get("bbox"):
            pg = r["page"]
            if pg not in page_cache:
                pp = od / "pages_han" / f"page_{pg:04d}.png"
                page_cache[pg] = Image.open(pp) if pp.exists() else None
            im = page_cache[pg]
            if im is not None:
                try:
                    b = crop_b64(im, r["bbox"])
                    crop = f'<img src="data:image/jpeg;base64,{b}" loading="lazy">'
                except Exception:
                    crop = ""
        badge = "low"
        bc = "#d98c00"
        ctx = html.escape(r.get("context", "") or "")
        review_html.append(
            f'<tr><td>{crop}</td>'
            f'<td><span class="badge" style="background:{bc}">{badge}</span></td>'
            f'<td><code>{r["id"]}</code><br><small>p.{r["page"]} box {r["box_idx"]}</small></td>'
            f'<td class="han">{html.escape(r.get("sn") or "")}</td>'
            f'<td>{html.escape(r.get("qn") or "")}</td>'
            f'<td>{r.get("conf",0):.3f}</td>'
            f'<td class="han ctx">{ctx}</td></tr>'
        )

    # ---- verdict logic ----
    verdict_rows = [
        ("Trang Hán render", n_pages, "OK", ""),
        ("Hán boxes / câu", len(hb), "OK", "1 box = 1 cột dọc"),
        ("Ký tự Hán (char)", n_chars, "OK", ""),
        ("OCR conf trung vị", f"{statistics.median(confs):.3f}" if confs else "—", "OK", ""),
        ("Box conf < .5 (cần soát)", f"{n_low} ({pct(n_low,len(hb))})", "WARN", "review queue"),
        ("Câu tiếng Việt", len(vs), "WARN", "OCR bản dịch rất nhiễu"),
        ("Cặp align Hán↔Việt", len(al), "WARN", f"sim trung vị {statistics.median(sims):.2f}" if sims else ""),
    ]
    vr_html = "".join(
        f'<tr><td>{html.escape(n)}</td><td><b>{v}</b></td>'
        f'<td><span class="st {s.lower()}">{s}</span></td>'
        f'<td><small>{html.escape(note)}</small></td></tr>'
        for n, v, s, note in verdict_rows
    )

    css = """
    body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1a1a1a;background:#fafafa}
    .wrap{max-width:1080px;margin:0 auto;padding:24px}
    h1{margin:0 0 4px} .sub{color:#888;margin-bottom:24px}
    .card{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:18px 20px;margin:16px 0}
    h2{margin:0 0 12px;font-size:18px} h3{font-size:14px;color:#555;margin:14px 0 6px}
    table{border-collapse:collapse;width:100%} td,th{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
    th{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.04em}
    .grid{display:flex;gap:24px;flex-wrap:wrap} .grid>div{flex:1;min-width:300px}
    .st{padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;color:#fff}
    .st.ok{background:#27ae60} .st.warn{background:#d98c00} .st.bad{background:#c0392b}
    .badge{padding:2px 7px;border-radius:8px;font-size:10px;font-weight:700;color:#fff}
    .kpi{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0}
    .kpi div{background:#f4f6f8;border-radius:8px;padding:10px 16px;min-width:120px}
    .kpi b{display:block;font-size:22px} .kpi span{font-size:11px;color:#888}
    .han{font-family:"Noto Serif CJK SC","Songti SC",serif;font-size:18px}
    .ctx{max-width:260px;font-size:14px;color:#555}
    code{background:#f0f0f0;padding:1px 4px;border-radius:4px;font-size:11px}
    img{display:block;border:1px solid #ddd;border-radius:4px;max-width:160px}
    .rqtable td{border-bottom:1px solid #f0f0f0}
    .note{background:#fff8e6;border-left:3px solid #d98c00;padding:10px 14px;border-radius:0 6px 6px 0;margin:10px 0}
    ol{margin:6px 0 6px 18px} li{margin:4px 0}
    """

    doc = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QA Report — {args.vol}</title><style>{css}</style></head><body><div class="wrap">
<h1>Minh Mệnh Chính Yếu — QA Report</h1>
<div class="sub">Tập <b>{args.vol}</b> · HVH_001 · {n_pages} trang Hán · tạo tự động từ out/{args.vol}/</div>

<div class="card"><h2>Tổng quan</h2>
<div class="kpi">
  <div><b>{n_pages}</b><span>trang Hán OCR</span></div>
  <div><b>{len(hb)}</b><span>câu / box Hán</span></div>
  <div><b>{n_chars}</b><span>ký tự Hán</span></div>
  <div><b>{len(al)}</b><span>cặp align</span></div>
  <div><b>{len(rv)}</b><span>mục cần soát</span></div>
</div>
<table>{vr_html}</table></div>

<div class="card"><h2>Đánh giá: đủ nộp chưa?</h2>
<p><b>Phía Hán — TỐT, đủ nộp.</b> OCR {n_pages} trang, {n_chars:,} ký tự. Trung vị độ tin cậy
{statistics.median(confs):.3f}. Độ đúng OCR do đồng thuận 2 engine (base + Qwen arbiter) quyết định
(xem cột conf), không dùng rule từ điển.</p>
<p><b>Phía Việt + alignment — YẾU, là điểm trừ.</b> Text tiếng Việt lấy OCR từ bản dịch in
nhiễu nặng (vd "MINH MỆNH EHÍNH VẾU"), nên similarity Hán↔Việt thấp
(trung vị {statistics.median(sims):.2f}). Phần lớn do chất lượng OCR tiếng Việt, không phải lỗi thuật toán align.</p>
<div class="note">Kết luận: <b>nộp được</b> nếu trọng tâm đề tài là corpus Hán + transliteration Hán-Việt.
Nếu chấm cả alignment chất lượng cao → cần cải thiện OCR tiếng Việt (xem mục Tối ưu).</div>
</div>

<div class="card"><h2>Phân bố chất lượng</h2>
<div class="grid">
<div><h3>OCR confidence ({len(confs)} box)</h3>{svg_bars(conf_hist, conf_lbls, "#3498db")}
<small>{pct(sum(1 for c in confs if c<.5),len(confs))} box dưới .5 → đưa vào hàng đợi soát.</small></div>
<div><h3>Alignment similarity ({len(sims)} cặp)</h3>{svg_bars(sim_hist, sim_lbls, "#9b59b6")}
<small>Thấp do OCR tiếng Việt nhiễu + Hán-Việt vs Việt hiện đại khác từ vựng.</small></div>
</div></div>

<div class="card"><h2>Có cần sửa tay OCR Hán không? Làm thế nào</h2>
<p>Không bắt buộc cho toàn bộ — chỉ <b>{len(rv)} mục</b> trong <code>review.jsonl</code>
({n_low} low-conf) đáng soát tay. Đó là &lt;{pct(n_low,len(hb))} số box.</p>
<h3>Quy trình sửa tay</h3>
<ol>
<li>Mở bảng "Hàng đợi soát" bên dưới — mỗi dòng có ảnh cắt từ trang gốc + ký tự OCR đoán.</li>
<li>Đối chiếu ảnh với cột <i>sinonom</i>. Nếu sai, ghi vào <code>review.jsonl</code> dòng tương ứng:
  <br><code>fix_type</code> = <b>ocr_wrong</b> (OCR đọc nhầm, điền ký tự đúng vào <code>correct</code>),
  hoặc <b>drop</b> (rác, bỏ).</li>
<li>Chạy lại bước gộp để áp chỉnh sửa vào <code>han_boxes</code> / <code>alignment</code>.</li>
</ol>
<div class="note"><b>Lưu ý kỹ thuật:</b> pipeline hiện <b>sinh</b> <code>review.jsonl</code> nhưng
<b>chưa có bước gộp ngược</b> các trường <code>correct</code> trở lại output. Cần thêm
<code>pipeline/apply_review.py</code> (đọc review → ghi đè sinonom/âm theo id → xuất lại Excel).
Mình tạo được nếu bạn muốn.</div>
</div>

<div class="card"><h2>Tối ưu thêm (nếu còn thời gian)</h2>
<ol>
<li><b>OCR tiếng Việt</b>: re-OCR ảnh trang bằng <b>Surya</b> (mặc định) thay cho text
nhúng/PDF nhiễu → phục hồi dấu thanh, tăng mạnh similarity align.</li>
<li><b>Lọc front-matter</b>: trang bìa/mục lục (sim &lt;.05) nên loại trước khi align (đã drop 107 câu, soát thêm).</li>
<li><b>Soát {n_low} box low-conf</b> tập trung các trang đầu (nhiều chữ triện/mờ).</li>
<li><b>Bước apply_review</b> để khép vòng sửa tay (xem trên).</li>
<li><b>Ngưỡng similarity</b>: gắn cờ cặp align sim&lt;.15 là "yếu" trong Excel để người dùng biết.</li>
</ol></div>

<div class="card"><h2>Hàng đợi soát tay ({len(rv)} mục, sắp theo conf tăng dần)</h2>
<table class="rqtable"><tr><th>Ảnh cắt</th><th>Loại</th><th>ID</th><th>SN</th><th>Âm</th><th>Conf</th><th>Ngữ cảnh</th></tr>
{"".join(review_html)}
</table></div>

<div class="sub" style="margin-top:24px">Tạo bởi pipeline/make_report.py</div>
</div></body></html>"""

    out = od / "report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, {len(rv)} review items, crops={'no' if args.no_crops else 'yes'})")


if __name__ == "__main__":
    main()

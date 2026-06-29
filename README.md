# Minh Mệnh Chính Yếu — HVB parallel Hán–Việt corpus pipeline

End-to-end pipeline for building a **parallel Sino-Vietnamese (Hán–Việt) corpus**
(track **HVB**: image input → OCR → sentence segmentation → alignment) from the
scanned *Minh Mệnh Chính Yếu* (明命政要) volumes.

Each volume is a scanned book whose layout is **Vietnamese (Quốc ngữ) translation
first, then the Hán (classical Chinese) original**. The two halves are the same
content, which is exactly what we align.

```
data/vol1.pdf ─► [1] split ─► VI pages ─► [2] re-OCR (Surya)+spell+segment ─► vi_sentences
                          └─► HÁN pages ─► [3] preprocess+OCR(vertical)+segment ─► han_* ─┐
                                                                                           │
                                    [4] bge-m3 align + S1∩S2 char check ◄─────────────────┘ ─► Excel
```

Fully offline — no API calls. The PDFs carry an embedded OCR layer (OCRmyPDF +
Tesseract `vie`) that is good enough to **classify** a page (Việt vs Hán) but
mangles the actual content — even on Quốc ngữ it drops tone marks / diacritics
("MINH MỆNH EHÍNH VẾU"), which wrecks the step-4 alignment. So step 2 **re-OCRs
the Vietnamese page images with Surya** (modern multilingual OCR that runs its
own line detection + recognition), recovering the diacritics at the source.
Classical Hán is written in **vertical columns right-to-left** and the embedded
layer renders it as Latin garbage — so for the Hán half that layer is used only
as a page-split signal, and step 3 re-OCRs from the images: detect boxes, then
recognise each column the vertical way (~70%). Residual character errors are
flagged/corrected offline by the **S1∩S2** rule in step 4.

Before OCRing a VI page, step 2 crops off the **top running-head band** (the
`QUYỂN n   <page-no>` line, or a lone page number) so Surya doesn't read it as a
sentence and inject noise into `vi_sentences` / the alignment. The crop is
conservative — it fires only on a single thin line high in the top margin that is
set off from the body by a wide gap, so content titles and chapter-opening pages
are never cut (`VIETNAMESE["crop_header"]`, on by default).

Step 1 also **trims the Vietnamese front matter** — the cover, half-title,
colophon and translator-credit pages that are Quốc ngữ (so they classify as VI)
but have no Hán counterpart and would force-match the Hán side. The translation
body cites the Hán original's leaf in brackets (`[1a]`, `[1b]`, a bare `[1]`, or
the spelled `[tờ 3b]`); front matter never does, so `vi_body_start` is the first
VI page bearing such a leaf marker that also reads as real body. Pages before it
are dropped and recorded in `split_manifest.json` (`vi_body_start`,
`vi_front_matter`). Validated on vol1–6 (`SPLIT["trim_front_matter"]`, on by
default).

## Layout

```
pipeline/
  config.py            all paths + thresholds (edit ID schema / DSG code here)
  common.py            logging, JSONL IO, ID schema, text helpers
  preprocess.py        image cleanup (deskew/denoise/binarize) before OCR
  build_dicts.py       build hanviet.csv + S1/S2 dicts from Unihan + samples
  step1_split_pdf.py   classify pages (VI/HÁN/plate/blank), split, trim VI front matter, render
  step2_vietnamese.py  re-OCR VI images (Surya) → spell-fix → underthesea
  vi_ocr.py            VI image OCR; crops the top running-head/page-number band
  step3_sinonom.py     preprocess → PaddleOCR detect + vertical recognise → segment
  step4_align.py       S1∩S2 char validation + review/OOV lanes + Excel
                       (sentence alignment runs in notebook ③, bge-m3)
assets/
  dicts/   Viet74K.txt, hanviet.csv, QuocNgu_SinoNom.dic (S2), SinoNom_Similar.dic (S1)
  raw/     Unihan_Readings.txt, Unihan_Variants.txt
out/<vol>/  all generated artifacts
```

## Data sources (all public)

| Resource | Source | Used for |
|---|---|---|
| âm Hán-Việt readings | Unicode **Unihan** `kVietnamese` (8.3k chars) | S2 + transliteration |
| visual-similar fallback | Unihan variant fields | S1 |
| `.dic` sample format | `khang3004/SinoNomViet_Transliteration_OCR` | S1/S2 seed + format |
| Vietnamese wordlist | `duyet/vietnamese-wordlist` (Viet74K) | page split + spell-fix |

> The full course `SinoNom_Similar.dic` / `QuocNgu_SinoNom.dic` are not public.
> We rebuild equivalents from Unihan and merge the public samples. **Drop the
> real course files into `assets/dicts/` (same format) to upgrade quality** —
> the code will use them automatically.
>
> Unihan `kVietnamese` covers ~8.3k chars and occasionally lists a Nôm reading
> instead of the Hán-Việt one. Fix individual chars in
> `assets/dicts/hanviet_overrides.csv` (`<char>,<âm Hán-Việt>`); `build_dicts`
> applies them as the primary reading. Seeded with common classical chars
> (何→hà, 虜→lỗ, 帝→đế, …).

## Run order

```bash
python -m pipeline.build_dicts                 # one-time: build dictionaries
python -m pipeline.step1_split_pdf  --vol vol1 # split + render page images
python -m pipeline.step2_vietnamese --vol vol1 # Vietnamese OCR (Surya) + segment
python -m pipeline.step3_sinonom    --vol vol1 # Hán OCR (vertical) + segment
python -m pipeline.step4_align      --vol vol1 # align + S1∩S2 + Excel
```

Output workbook: `out/vol1/HVH_001_alignment.xlsx` with sheets
`boxes` (spec layout: ID · Image box · SinoNom char · Âm Hán Việt · Nghĩa thuần Việt),
`sentence_alignment` (m-n pairs + similarity), `char_validation` (S1∩S2 colours).

## ID schema

`DSG_fff.ccc.ppp.ss` — set `ID_SCHEMA` in `config.py`. Default `HVH_001`
(**H**istory · **V**ietnam · **H**án-genre · file `001`). Replace `fff` with the
`DSG_fff` code the instructor assigns; `ccc/ppp/ss` are auto-numbered
(chapter/page/sentence-or-box).

## Colab

Current flow is split across three notebooks, each handing off one zip per volume:
`Minh_Menh_1_VI_OCR_Surya_Colab.ipynb` (VI side, Surya → `out_zips/<vol>.zip`) →
`Minh_Menh_2_Han_OCR_Colab.ipynb` (Hán PaddleOCR + Qwen-VL consensus →
`ocr_zips/<vol>.zip`) → `Minh_Menh_3_Align_Colab.ipynb` (bge-m3 sentence alignment
of the VI + corrected Hán → `output/<vol>.zip`). They are split because Surya, the
paddle/qwen OCR stack, and bge-m3 conflict over Pillow, so each runs in its own
runtime. Run on a Colab Pro runtime (GPU recommended). `Minh_Menh_Pipeline_Colab.ipynb`
is the older all-in-one (deprecated).

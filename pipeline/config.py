"""Central configuration for the Minh Mệnh Chính Yếu HVB pipeline.

Track: HVB — parallel Sino-Vietnamese (Hán–Việt) corpus, image input:
OCR -> sentence segmentation -> alignment.

All paths are resolved relative to the project root so the same config works
locally and on Google Colab (where you typically clone/mount into /content).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = parent of this `pipeline/` folder.
ROOT = Path(os.environ.get("MMCY_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = ROOT / "data"            # input PDFs: vol1.pdf ... vol6.pdf
ASSETS_DIR = ROOT / "assets"
DICTS_DIR = ASSETS_DIR / "dicts"
RAW_DIR = ASSETS_DIR / "raw"
OUT_DIR = ROOT / "out"              # all generated artifacts land here

# Dictionary files (see build_dicts.py)
VIET_WORDLIST = DICTS_DIR / "Viet74K.txt"
UNIHAN_READINGS = RAW_DIR / "Unihan_Readings.txt"
UNIHAN_VARIANTS = RAW_DIR / "Unihan_Variants.txt"
# KanjiDictVN (trungnt2910/KanjiDictVN) — KANJIDIC2/EDRDG char set with curated
# âm Hán-Việt readings. Reputable base (EDRDG, CC BY-SA) that fills the chars the
# sparse Unihan kVietnamese field misses. Cached here at build time.
KANJIDICTVN_FILES = (RAW_DIR / "kanjidictvn_1.json", RAW_DIR / "kanjidictvn_2.json")
KANJIDICTVN_URLS = (
    "https://raw.githubusercontent.com/trungnt2910/KanjiDictVN/master/out_vn/kanji_bank_1.json",
    "https://raw.githubusercontent.com/trungnt2910/KanjiDictVN/master/out_vn/kanji_bank_2.json",
)

HANVIET_CSV = DICTS_DIR / "hanviet.csv"                 # sinonom_char, am_han_viet, all_readings
HANVIET_SUPPLEMENT = DICTS_DIR / "hanviet_supplement.csv"  # curated fills for kVietnamese gaps
QUOCNGU_SINONOM_DIC = DICTS_DIR / "QuocNgu_SinoNom.dic"  # S2: quoc_ngu -> sinonom chars
SINONOM_SIMILAR_DIC = DICTS_DIR / "SinoNom_Similar.dic"  # S1: sinonom -> visually similar chars
VI_CORRECTIONS_CSV = DICTS_DIR / "vi_corrections.csv"    # domain OCR fixes (step 2): wrong,correct,note

# --------------------------------------------------------------------------- #
# Step 1 — PDF splitting (Vietnamese front matter vs. Hán original)
# --------------------------------------------------------------------------- #
# The PDFs are scanned images carrying a noisy embedded OCR text layer. That
# layer is good enough to *classify* a page (Vietnamese vs. Hán) even though it
# is useless for extracting real Hán content (Hán renders as Latin garbage).
SPLIT = dict(
    # A page with fewer than this many word tokens in its text layer carries no
    # usable text. It is then split further by ink coverage (see plate_ink_ratio)
    # into a truly-blank separator page vs. an image-only Hán plate (e.g. the
    # woodblock title/cover 明命政要), which must be kept in the Hán half.
    min_tokens=10,
    # Fraction of dark pixels (rasterised at plate_ink_dpi) above which a
    # text-less page is treated as an image PLATE rather than a blank separator.
    # Truly blank pages measure ~0.00; Hán woodblock plates measure ~0.12-0.21.
    plate_ink_ratio=0.02,
    plate_ink_dpi=80,
    # Fraction of tokens carrying a Vietnamese diacritic. This is the reliable
    # signal: the embedded text layer was produced by a Vietnamese OCR engine,
    # so real Vietnamese pages are densely accented (~0.8) while Hán pages OCR
    # to mostly-ASCII garbage (~0.4). Word-overlap is NOT reliable (short Hán
    # garbage tokens coincidentally match Vietnamese syllables).
    vi_diac_ratio=0.65,
    # The Hán section is only confirmed once we see a run of MORE THAN this many
    # consecutive Hán pages (blank pages do not break the run, but cannot start
    # it). This filters chapter-title / image-only false positives.
    han_confirm_run=3,
    # DPI used when rasterising pages to PNG for the OCR steps.
    render_dpi=300,
    # Trim the Vietnamese FRONT MATTER (cover, half-title, colophon, translator-
    # credit pages) that has no Hán counterpart and otherwise force-matches the
    # Hán side. The translation body cites the Hán original's leaf number in
    # brackets ("[1a]", "[1b]", bare "[1]", or spelled "[tờ 3b]") — front matter
    # never does. vi_body_start = first VI page bearing such a leaf marker AND
    # looking like real body (dense OR carrying a chapter/reign keyword). Pages
    # before it are dropped from the VI half. Validated on vol1-6 (body starts at
    # the leaf marker in every volume). Fallback if no marker OCRs cleanly: the
    # last "… dịch của <NAME>" half-title page → next dense page.
    trim_front_matter=True,
    leaf_marker_regex=r"[\[\(]\s*(?:t[ờo]\s*)?\d{1,3}\s*[-–\s]*[abAB]?\s*[\]\)]",
    body_min_tokens=120,      # a body page is dense unless it carries a keyword
    # OCR-tolerant guards: a leaf-marker page also counts as body if it names a
    # chapter / reign even when short (vol6 chapter-opening is ~100 tokens).
    body_keyword_regex=r"QUY[ỂỀE]N|M[ỆE]NH",
    halftitle_regex=r"d[ịi][cz]h\s+c[ủuú]a",   # "(bản) dịch của <translator>"

    # Back-matter trim — symmetric to the front-matter trim above. Some volumes
    # append a table-of-contents, a name/place index, and a publisher's book
    # catalogue / bibliography AFTER the body. These have no Hán counterpart and
    # are full of proper nouns / foreign titles, so they bury the VI review queue
    # in false high-OOV flags. vi_body_end = last running-body page; everything
    # after it is dropped, but ONLY when that trailing block carries a back-matter
    # anchor (so volumes that end on body are never trimmed). Detection runs on
    # the embedded text layer (pre-OCR). Validated on vol1-6: vol3 (drop ≥p295),
    # vol5 (≥p215), vol6 (≥p273) trim; vol1/2/4 keep all. See find_vi_body_end.
    trim_back_matter=True,
    index_line_regex=r"[^\s:：]+\s*[:：]\s*\d",   # "<name/title> : <page no.>"
    index_line_frac=0.5,        # page is an index if >= this frac of lines match
    short_line_max_tokens=4,    # a "short line" (index / ToC entries are short)
    short_body_frac=0.35,       # >= this frac short lines => not running prose
    min_body_lines=10,          # a body page has at least this many text lines
    # Section headers / phrases that only appear in back-matter (OCR-tolerant).
    back_matter_anchor_regex=(
        r"TH[ƯU]\s*[-–]?\s*M[ỤU]C"                       # THƯ MỤC (bibliography)
        r"|NGUYÊN[-\s]*TÁC"                              # Nguyên-tác
        r"|T[ỦU][-\s]*SÁCH"                              # Tủ-sách (catalogue)
        r"|B[ẢA]N\s*D[ỊI]CH\s*C[ỦU]A"                    # Bản dịch của
        r"|BI[ỂẾEỀ]U\s*K[ÊẾE]"                           # Biểu kê (index header)
        r"|B[ẢA]NG\s*TRA"                                # Bảng tra
        r"|H[ẾEỀ]T\s*T[ẬAẤ]P"                            # HẾT TẬP (volume end)
        r"|\b\d{1,2}\s*[.,]?\s*[a-z]?\s*/\s*[A-Z]{2}\b"  # catalogue codes 12,a/CV 01/KV
    ),
)

# --------------------------------------------------------------------------- #
# Step 2 — Vietnamese side (Quốc ngữ)
# --------------------------------------------------------------------------- #
# The Vietnamese translation text is read by re-OCRing the rendered page images
# with Surya, a modern multilingual transformer OCR. It recovers the diacritics/
# tone marks and the b/h ascender that flatten on this 1970s reprint
# ("tbần"->"thần", "Trằm"->"Trẫm") at the source — the embedded Tesseract layer
# mangles them ("MINH MỆNH EHÍNH VẾU") and wrecks step-4 alignment. Surya runs
# its OWN line detection (no PaddleOCR detector here), and conflicts with paddle
# over Pillow — run the VI side in a clean Surya runtime (notebook ①).
VIETNAMESE = dict(
    min_line_height=8,        # drop detection boxes thinner than this (px @ render_dpi)
    row_tol=12,               # y-centre tolerance (px) for grouping boxes into a row
    cache_images=True,        # also save rendered VI pages to out/<vol>/pages_vi/
    # Strip the running-head / page-number band at the TOP of each VI page before
    # OCR. The body pages carry a one-line head ("QUYỂN 2     83") set off from the
    # text by a wide gap; left in, Surya reads it as a sentence ("QUYỂN 2 83") and
    # injects noise into vi_sentences / the alignment. We crop the first text band
    # only when it is a single thin line high in the top margin AND separated from
    # the body by a gap clearly larger than a normal line gap — so a content title
    # (e.g. "NĂM MINH MỆNH THỨ CHÍN") or a chapter-opening page is never cut.
    crop_header=True,
    header_zone=0.18,         # first band must start within this fraction of page height
    header_gap_factor=1.4,    # gap below header >= this * median line gap to qualify
    header_max_lines=1.8,     # header band height <= this * median line height (single line)

    # Dictionary-based spell normalisation: only auto-correct an out-of-vocab
    # token when a single in-vocab candidate exists within this edit distance.
    spell_max_edit_distance=1,
    spell_min_token_len=2,    # don't try to "fix" tokens shorter than this
    # Three normalisation passes always run in step 2 (raw_text is kept):
    #  - Domain correction map (assets/dicts/vi_corrections.csv): deterministic
    #    phrase-level fixes for reign titles / institution names the generic
    #    edit-distance fixer can't reach. Applied first.
    #  - Confusable-character fixer: targeted single-char OCR substitutions
    #    (e.g. b<->h) applied to an OOV token only when EXACTLY ONE variant is in
    #    vocab. Kills the ambiguity plain edit-distance can't ("tbần" is edit-1
    #    from thần/trần/tần — confusion rule narrows it to "thần").
    #  - Dictionary spellcheck: see spell_* params above.
)

# --------------------------------------------------------------------------- #
# Image preprocessing (step 3 Hán OCR, before recognition)
# --------------------------------------------------------------------------- #
# Old scans benefit a lot from cleanup before the Hán OCR in step 3. (Step 2 does
# its own image OCR with Surya, which handles the VI pages separately.)
PREPROCESS = dict(
    grayscale=True,
    deskew=True,             # straighten slightly rotated scans
    denoise=True,            # remove speckle
    binarize="adaptive",     # "adaptive" | "otsu" | None
    upscale_min_height=0,    # upscale page if shorter than this (0 = off)
)

# --------------------------------------------------------------------------- #
# Step 3 — Hán / SinoNom side
# --------------------------------------------------------------------------- #
# Classical Hán is written in VERTICAL columns, right-to-left. A horizontal
# recogniser reads it almost randomly (~10% chars); recognising each column the
# right way is essential (~70%). PaddleOCR is the single OCR engine here: it
# detects the column boxes, we reconstruct reading order, then recognise each
# column by rotating the crop 90°. Residual errors are flagged/corrected offline
# by the S1∩S2 rule in step 4.
SINONOM = dict(
    preprocess=True,
    paddle_lang="chinese_cht",   # traditional Chinese fits woodblock prints
    # column clustering tolerance (fraction of median box width).
    column_tol=0.6,
    # Classical Hán is frequently unpunctuated. If a page has fewer than this
    # fraction of sentence-final marks we treat each column/box as a sentence.
    min_punct_ratio=0.01,
    sentence_punct="。！？；",
    clause_punct="，、：",
)

# --------------------------------------------------------------------------- #
# Step 4 — Alignment + export
# --------------------------------------------------------------------------- #
ALIGN = dict(
    # DP alignment search: allowed merge patterns (han:viet).
    # The Hán side is segmented per COLUMN (unpunctuated woodblock, ~12 chars),
    # while the VI side is full sentences (~27 words) from underthesea — so a
    # single VI sentence usually spans SEVERAL Hán columns. Allow up to 5 Hán
    # columns ↔ 1 VI sentence (and 1↔2) instead of forcing a 2:2 ceiling.
    merge_modes=((1, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1),
                 (1, 0), (0, 1), (2, 2)),
    sim_threshold=0.30,        # below this, treat as gap rather than match
    # Lexical cross-check (a second opinion on each aligned pair). Vietnamese keeps
    # a large Sino-Vietnamese vocabulary, so a CORRECT pair shares tokens between
    # the âm Hán-Việt reading and the translation ("bản triều", "trung thần") even
    # when the neural model underrates classical văn-ngôn. A pair is flagged
    # `suspect` only when BOTH signals are weak — one strong signal (neural OR
    # lexical) clears it, so neither method's blind spot raises a false alarm alone.
    suspect_lexical=0.05,
    # Drop VI "sentences" that are front-matter / OCR garbage before aligning
    # (library stamps, title pages, publisher lines). A sentence is kept only if
    # at least this fraction of its alphabetic tokens are real Vietnamese words.
    min_vi_invocab_ratio=0.5,
    min_vi_tokens=2,
)

# --------------------------------------------------------------------------- #
# Step 3b — Hán OCR consensus (clean characters BEFORE alignment)
# --------------------------------------------------------------------------- #
# PaddleOCR ("base") mis-reads blurry / rare woodblock characters with high
# confidence. A vision-language model re-reads each column image and arbitrates;
# this measurably fixes real errors (e.g. the title 明命政要 base read as 明命政安饰).
# Method + model I/O live in notebook ② via pipeline.han_consensus; thresholds here.
#
# `spec` (a SinoNom-specialised PP-OCRv5 rec) agrees with base only ~0.4% on this
# corpus and does NOT decide the vote in qwen_arbiter mode, so it is OFF by
# default. The real cost lever is gating the (expensive) VLM on base confidence:
# only re-read columns the base engine was unsure about.
CONSENSUS = dict(
    # engines
    spec_repo="MinhDS/Fine-tuned-PaddleOCRv5",   # HF Space: Hán-Nôm fine-tuned rec
    spec_name="PP-OCRv5_server_rec",
    use_spec=False,            # specialist ~0.4% agreement + not decisive -> off (set True to mirror NB4)
    qwen_model="Qwen/Qwen2.5-VL-7B-Instruct",
    load_4bit=True,            # 4-bit (~6 GB) so it co-resides with bge-m3 on a T4
    vote_mode="qwen_arbiter",  # "qwen_arbiter" (base==qwen keep base; else trust qwen) | "majority"
    # which columns get the VLM:
    #   "full"    — every column (most accurate, ~3h/vol; use to re-tune)
    #   "cascade" — only the columns below worth re-reading (production default):
    #               base conf < qwen_conf_gate, OR (use_spec and base != spec).
    vlm_mode="cascade",
    qwen_conf_gate=0.90,       # cascade: skip the VLM on columns base read with conf >= this
    # crop / decode
    spec_rotate="cw",          # vertical column -> rec orientation: "none" | "cw" | "ccw"
    crop_inset=4,              # px trimmed off each column bbox to drop the box border
    upscale=2,                 # upscale the crop before recognition
    max_new_tok=64,
)

# --------------------------------------------------------------------------- #
# Review queues — flag items a human should check. Split by stage:
#   vi_review.jsonl (high_oov, P1 step 2) + han_review.jsonl (RED + low_conf, P2 step 3).
# --------------------------------------------------------------------------- #
REVIEW = dict(
    # A box whose OCR confidence is below this is flagged as "đọc mờ" (low_conf).
    # RED chars are flagged regardless of confidence (confident-but-wrong errors).
    conf_threshold=0.5,
    # A Vietnamese sentence whose out-of-vocabulary token rate is at/above this
    # (and has >= min_oov_tokens tokens) is flagged high_oov: the VI OCR likely
    # read non-words. Soft signal — proper nouns / Hán-Việt terms inflate it — so
    # it is a separate review lane from the Hán RED/low_conf flags, never a drop.
    oov_threshold=0.2,    # Surya is clean (vol1 median OOV ~0.03); 0.2 isolates
                          # the genuinely garbled lines without burying the queue.
    min_oov_tokens=4,
)

# --------------------------------------------------------------------------- #
# ID schema — DSG_fff.ccc.ppp.ss  (see SinoNom_OCR_TransliterationAlignment.pdf)
# --------------------------------------------------------------------------- #
# Minh Mệnh Chính Yếu: History domain, Vietnamese history.
#   D = H (History)
#   S = V (sub-domain: Vietnamese history)   [adjust to whatever the instructor assigns]
#   G = H (Genre: Hán, vertical, woodblock-print)
#   fff = 001 (file id assigned by instructor; placeholder until provided)
ID_SCHEMA = dict(
    domain="H",       # D
    subdomain="V",    # S
    genre="H",        # G
    file_id="001",    # fff  -> instructor-provided "DSG_fff"
)
# => prefix "HVH_001". chapter(ccc)/page(ppp)/sentence(ss) are auto-numbered.

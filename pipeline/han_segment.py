"""Hán sentence segmentation — auto-punctuation of unpunctuated woodblock text.

Woodblock prints carry NO punctuation, so step 3 falls back to "one column == one
sentence". But a classical sentence flows across several columns (a VI sentence maps
to ~8 Hán columns), so column-as-sentence over-segments the Hán side ~3x and wrecks
the bge sentence alignment. This module re-segments the CLEAN (post-consensus) Hán
character stream into real sentences using an auto-punctuation model.

Split scope is the **whole chapter**: all pages' main columns are concatenated in
reading order into ONE character stream, punctuated, then split — so a sentence that
runs across a column OR page boundary stays whole (nothing is dropped). Because a
sentence can now span pages, ``box_ids`` are the GLOBAL box ``id`` strings (not
page-local indices); each sentence's ``page`` is where its first box sits. draw_boxes
resolves a box id straight to its (page, bbox). step4 / the aligner key on the
sentence id + text and are unaffected.

The heavy model I/O (a token-classification punctuator, e.g.
``raynardj/classical-chinese-punctuation-guwen-biaodian``) lives in notebook ②; this
module is dependency-free (stdlib only) so it stays testable. The notebook passes in
``labels_fn(text) -> list[str]`` returning, for each input character, the punctuation
mark predicted to follow it ("" for none) — it slides a fixed window with overlap over
the long stream internally, so no sentence is cut at a model-window boundary. We split
on sentence-final marks and map each sentence back to the columns it covers.

雙行 (interlinear-note) columns are NOT punctuated — their reading order is only a
geometric guess (``is_dbl``). Each 雙行 column is emitted as its own sentence, tagged
``is_dbl=True`` so the aligner can isolate / down-weight it.

Two cleanup passes run on the corrected boxes BEFORE re-segmentation:

``drop_banxin``   — every woodblock leaf carries a 版心 strip outside the text frame
                    (book title 明命政要 + section + 卷/leaf number). The detector
                    sometimes picks it up as a normal column and its garbled OCR
                    ("月令文要法度工三") lands INSIDE cross-page sentences when the
                    chapter stream is concatenated. Audit (vol1–6): 24–128 such boxes
                    per volume, most with conf ≥ 0.5 so the review queue never sees
                    them. Geometry alone over-fires (a genuine last column can sit at
                    the frame's left edge — vol3 p316 知幾倍可見), so a box is dropped
                    only when it is short AND in an edge strip AND most of its chars
                    come from the closed 版心 vocabulary (title/section/numerals plus
                    their common OCR misreads).

``normalize_s2t`` — the recogniser leaks simplified variants (~2.5% of chars: 赏数则
                    员学...) into this traditional-script corpus. Deterministic
                    simplified→traditional mapping built from the observed chars;
                    ambiguous simplifications (后/发/干/里…) are left untouched.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

# sentence-final marks the punctuator may emit (。！？；). ； (semicolon) is included:
# in classical prose it typically closes a full clause that aligns like a sentence.
SENT_MARKS = "。！？；"

# --------------------------------------------------------------------------- #
# 版心 (leaf-title strip) filter
# --------------------------------------------------------------------------- #
# Closed vocabulary of the 版心 line: title 明命政要 + 卷之 + Chinese numerals +
# the section names of this book (敬天法祖 / 孝治 / 勤政 / 愛民 / 法度 / 制兵 /
# 慎刑 / 武功 / 求賢 / 廣言路 …) + the recogniser's habitual misreads of those
# glyphs on the fold (政→文/攻/正, 命→令, 明→月, 要→更/男/汝 …).
BANXIN_CHARS = set(
    "明命政要卷之上下止"
    "一二三四五六七八九十百千"
    "敬天法祖孝治勤政愛民制兵慎刑揆文奮武功求賢審官重農崇儉言路度廣"
    "月令更汝男史頌攻正女"
)

# Extra misreads seen ONLY at the binding fold, where the cut-off title sliver is
# blurred (明→日/目/旦, 命→合/名, 政→正/直, 要→西/區, 卷→老/勇/光, digits→
# 石/亞/川/口/豆/里/乙/午/具/楼/真/出/先/右/台). Applied only to the right-edge
# tests: these are common content chars, so counting them near the LEFT border
# would eat real last-column tails.
BANXIN_FOLD_CHARS = BANXIN_CHARS | set("日台亞石目右川旦午乙口豆里真老出合勇光直先楼西名具區")


def drop_banxin(boxes: list[dict], x_frac: float = 0.16, right_frac: float = 0.97,
                max_chars: int = 12, min_ratio: float = 0.6,
                soft_frac: float = 0.90, min_conf: float = 0.75,
                ) -> tuple[list[dict], list[dict]]:
    """Split ``boxes`` into (kept, dropped-版心). Non-destructive; log the drops.

    A box is 版心 when its text is short (``<= max_chars``), it sits in an edge
    zone, and it *looks* like leaf-title debris there. The charset test is what
    protects genuine content columns at the frame edge (vol3 p316 知幾倍可見).

    W is the VOLUME-level 95th percentile of box right edges — a stable estimate of
    the text frame's right border. It must NOT be per-page: on a sparse page the
    first (rightmost) column defines the page max itself, so any per-page right-edge
    test tautologically fires on real content (year headers 明命四年, section
    headings 勤政第七, a lone carried-over 之). Zones against W (x1 = box left):
      left       — ``x1 <= x_frac*W``: the full 版心 line, outside the left border.
                   Charset test only (BANXIN_CHARS).
      hard right — ``x1 >= right_frac*W``: beyond the content frame. Fold charset
                   OR ``conf < min_conf`` — the sliver is blurred, so real content
                   here is essentially only high-conf carried-over columns
                   (vol1 p371 治之及 conf 1.0, p323 法祖第二 conf .99).
      soft right — ``x1 >= soft_frac*W``: the first (rightmost) column's territory,
                   full of real page-opening content. Drop only on fold charset
                   AND low conf together (明月西楼三 conf .17); either signal alone
                   spares real text (米將盡 conf .71 ratio 0; 法祖第二 ratio .75
                   conf .99).
    """
    xs = sorted(b["bbox"][2] for b in boxes)
    if not xs:
        return list(boxes), []
    w = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
    # Real era-date lines (明命八年 …) start a new entry and can land in the last
    # (leftmost) column — they are content, never 版心, despite being title-charset.
    era_year = re.compile(r"^(明命|嘉隆)[〇一二三四五六七八九十百元]{0,4}年$")
    kept, dropped = [], []
    for b in boxes:
        text = b["sinonom"]
        x1 = b["bbox"][0]
        conf = b.get("conf", 1.0)
        if not text or len(text) > max_chars or era_year.match(text):
            kept.append(b)
            continue
        ratio = sum(1 for c in text if c in BANXIN_CHARS) / len(text)
        fold = sum(1 for c in text if c in BANXIN_FOLD_CHARS) / len(text)
        # a lone char can be a sentence tail spilling into the last column
        # (vol3 p406 治): only trust the extreme strip for single chars.
        left_frac = 0.10 if len(text) == 1 else x_frac
        if ((x1 <= left_frac * w and ratio >= min_ratio)
                or (x1 >= right_frac * w and (fold >= min_ratio or conf < min_conf))
                or (x1 >= soft_frac * w and fold >= min_ratio and conf < min_conf)):
            dropped.append(b)
        else:
            kept.append(b)
    return kept, dropped


# --------------------------------------------------------------------------- #
# simplified → traditional normalisation
# --------------------------------------------------------------------------- #
# Only unambiguous mappings (one traditional form in classical usage). Built from
# the simplified chars actually observed in the vol1–6 OCR output plus the rest of
# the common table. Ambiguous ones (后/發-髮 发/干/几/里/云/谷/斗…) are excluded.
S2T = {
    '数': '數', '则': '則', '举': '舉', '员': '員', '赏': '賞', '给': '給', '风': '風',
    '减': '減', '别': '別', '为': '為', '当': '當', '学': '學', '经': '經', '问': '問',
    '闻': '聞', '开': '開', '关': '關', '门': '門', '间': '間', '见': '見', '说': '說',
    '读': '讀', '书': '書', '长': '長', '东': '東', '车': '車', '马': '馬', '鸟': '鳥',
    '岛': '島', '点': '點', '党': '黨', '权': '權', '办': '辦', '协': '協', '单': '單',
    '双': '雙', '变': '變', '边': '邊', '达': '達', '迁': '遷', '过': '過', '还': '還',
    '进': '進', '运': '運', '连': '連', '远': '遠', '违': '違', '韩': '韓', '鲜': '鮮',
    '丽': '麗', '历': '歷', '厉': '厲', '励': '勵', '医': '醫', '区': '區', '华': '華',
    '实': '實', '宁': '寧', '广': '廣', '庆': '慶', '应': '應', '庙': '廟', '库': '庫',
    '废': '廢', '录': '錄', '绳': '繩', '纲': '綱', '纪': '紀', '约': '約', '级': '級',
    '纸': '紙', '细': '細', '织': '織', '终': '終', '结': '結', '绝': '絕', '统': '統',
    '丝': '絲', '绿': '綠', '网': '網', '罗': '羅', '罚': '罰', '贤': '賢', '责': '責',
    '贮': '貯', '购': '購', '贵': '貴', '贷': '貸', '费': '費', '贺': '賀', '资': '資',
    '赋': '賦', '赐': '賜', '赛': '賽', '赞': '贊', '军': '軍', '农': '農', '写': '寫',
    '况': '況', '凑': '湊', '击': '擊', '刘': '劉', '刚': '剛', '创': '創', '务': '務',
    '动': '動', '劳': '勞', '势': '勢', '汇': '匯', '汉': '漢', '汤': '湯', '兴': '興',
    '旧': '舊', '优': '優', '伤': '傷', '价': '價', '众': '眾', '体': '體', '余': '餘',
    '侠': '俠', '侧': '側', '俭': '儉', '请': '請', '诸': '諸', '诚': '誠', '话': '話',
    '语': '語', '谓': '謂', '议': '議', '译': '譯', '试': '試', '诗': '詩', '词': '詞',
    '该': '該', '详': '詳', '诣': '詣', '误': '誤', '谁': '誰', '调': '調', '谅': '諒',
    '谈': '談', '谋': '謀', '谢': '謝', '谕': '諭', '讹': '訛', '设': '設', '访': '訪',
    '证': '證', '评': '評', '识': '識', '红': '紅', '纯': '純', '纳': '納', '纷': '紛',
    '纵': '縱', '练': '練', '组': '組', '绅': '紳', '绍': '紹', '绎': '繹', '缘': '緣',
    '缠': '纏', '县': '縣', '严': '嚴', '丧': '喪', '个': '個', '丰': '豐', '临': '臨',
    '义': '義', '乌': '烏', '乐': '樂', '乡': '鄉', '买': '買', '乱': '亂', '争': '爭',
    '亏': '虧', '亚': '亞', '产': '產', '亲': '親', '亿': '億', '仅': '僅', '从': '從',
    '仑': '崙', '仓': '倉', '仪': '儀', '们': '們', '伦': '倫', '伪': '偽', '传': '傳',
    '倾': '傾', '偿': '償', '储': '儲', '兑': '兌', '内': '內', '冈': '岡', '册': '冊',
    '冯': '馮', '冲': '沖', '决': '決', '冻': '凍', '净': '淨', '凉': '涼', '凤': '鳳',
    '凭': '憑', '凯': '凱', '刍': '芻', '划': '劃', '剑': '劍', '剧': '劇', '劝': '勸',
    '劲': '勁', '勋': '勳', '匀': '勻', '岁': '歲', '狱': '獄', '脉': '脈', '静': '靜',
    '带': '帶', '绵': '綿', '缓': '緩', '编': '編', '缴': '繳', '职': '職', '聪': '聰',
    '肃': '肅', '胜': '勝', '腊': '臘', '舆': '輿', '舰': '艦', '艰': '艱', '节': '節',
    '荐': '薦', '药': '藥', '蓝': '藍', '虑': '慮', '虽': '雖', '蚕': '蠶', '蛮': '蠻',
    '补': '補', '衔': '銜', '装': '裝', '规': '規', '视': '視', '览': '覽', '觉': '覺',
    '誉': '譽', '计': '計', '订': '訂', '讨': '討', '让': '讓', '讯': '訊', '记': '記',
    '讲': '講', '许': '許', '论': '論', '讼': '訟', '讽': '諷', '诀': '訣', '奖': '獎',
    '将': '將', '尔': '爾', '尝': '嘗', '尧': '堯', '尽': '盡', '层': '層', '属': '屬',
    '岭': '嶺', '峡': '峽', '币': '幣', '帅': '帥', '师': '師', '帐': '帳', '归': '歸',
    '彻': '徹', '径': '徑', '忆': '憶', '怀': '懷', '态': '態', '总': '總', '恒': '恆',
    '恼': '惱', '悬': '懸', '惊': '驚', '惧': '懼', '惨': '慘', '慑': '懾', '忧': '憂',
    '战': '戰', '抚': '撫', '护': '護', '报': '報', '担': '擔', '拟': '擬', '择': '擇',
    '挂': '掛', '挚': '摯', '挥': '揮', '损': '損', '摄': '攝', '敌': '敵', '斋': '齋',
    '断': '斷', '无': '無', '旷': '曠', '显': '顯', '晋': '晉', '晓': '曉', '昼': '晝',
    '术': '術', '机': '機', '杀': '殺', '杂': '雜', '条': '條', '来': '來', '杨': '楊',
    '构': '構', '枢': '樞', '标': '標', '栏': '欄', '树': '樹', '样': '樣', '档': '檔',
    '桥': '橋', '检': '檢', '楼': '樓', '榄': '欖', '欢': '歡', '钦': '欽', '殁': '歿',
    '残': '殘', '殒': '殞', '毁': '毀', '气': '氣', '汹': '洶', '沟': '溝', '泞': '濘',
    '泪': '淚', '泽': '澤', '洁': '潔', '浊': '濁', '测': '測', '济': '濟', '浑': '渾',
    '浓': '濃', '涛': '濤', '涝': '澇', '润': '潤', '涧': '澗', '涌': '湧', '渊': '淵',
    '渐': '漸', '滞': '滯', '满': '滿', '滥': '濫', '滨': '濱', '潜': '潛', '灾': '災',
    '灿': '燦', '炼': '煉', '烂': '爛', '烛': '燭', '烦': '煩', '焕': '煥', '爱': '愛',
    '牍': '牘', '牵': '牽', '状': '狀', '犹': '猶', '独': '獨', '狭': '狹', '猎': '獵',
    '兽': '獸', '献': '獻', '玛': '瑪', '环': '環', '现': '現', '琼': '瓊', '瑶': '瑤',
    '疴': '痾', '症': '癥', '痒': '癢', '疗': '療', '盘': '盤', '监': '監', '盖': '蓋',
    '盗': '盜', '积': '積', '称': '稱', '稳': '穩', '穑': '穡', '窃': '竊', '窜': '竄',
    '窝': '窩', '竖': '豎', '笔': '筆', '筑': '築', '签': '簽', '简': '簡', '类': '類',
    '粜': '糶', '粮': '糧', '绌': '絀', '绢': '絹', '绣': '繡', '继': '繼', '绩': '績',
    # Leftovers found when auditing the NB2.5 output (~500 occurrences the first
    # dict missed; 须 taken as 須 — the 鬚 'beard' sense never occurs here):
    '须': '須', '国': '國', '陈': '陳', '题': '題', '张': '張', '阁': '閣',
    '辞': '辭', '馆': '館', '确': '確', '万': '萬', '龙': '龍', '与': '與',
    '颖': '穎',
}
_S2T_TABLE = str.maketrans(S2T)


def to_traditional(text: str) -> str:
    """Map leaked simplified chars to their traditional forms (unambiguous only)."""
    return text.translate(_S2T_TABLE)


def normalize_s2t(boxes: list[dict],
                  transliterate: Callable[[str, dict], str] | None = None,
                  hanviet: dict | None = None) -> int:
    """In-place s2t on every box's ``sinonom``; refresh ``am_han_viet`` when the
    transliterator is provided. Returns the number of chars converted."""
    changed = 0
    for b in boxes:
        new = to_traditional(b["sinonom"])
        if new != b["sinonom"]:
            changed += sum(1 for a, c in zip(b["sinonom"], new) if a != c)
            b["sinonom"] = new
            if transliterate is not None and hanviet is not None:
                b["am_han_viet"] = transliterate(new, hanviet)
    return changed


def _split_stream(chars: list[str], marks: list[str], sent_marks: str) -> list[tuple[int, int]]:
    """Return [start, end) spans over `chars`, breaking after any sentence-final mark.

    `marks[k]` is the punctuation predicted to follow `chars[k]` ("" for none). A
    trailing run with no closing mark is emitted as a final sentence.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(chars)
    for k in range(n):
        if marks[k] and marks[k] in sent_marks:
            spans.append((start, k + 1))
            start = k + 1
    if start < n:
        spans.append((start, n))
    return spans


def resegment_boxes(han_boxes: list[dict],
                    labels_fn: Callable[[str], list[str]],
                    transliterate: Callable[[str, dict], str],
                    hanviet: dict,
                    make_id: Callable[[int, int, int], str],
                    sent_marks: str = SENT_MARKS) -> list[dict]:
    """Rebuild han_sentences rows from (corrected) han_boxes via auto-punctuation.

    Per chapter: concatenate every non-雙行 column's chars — across all pages, in
    (page, box_idx) reading order — into ONE stream, get per-char marks from
    ``labels_fn``, split on sentence-final marks. A sentence may span columns and
    pages; nothing is dropped. Each 雙行 column stays its own tagged sentence.

    Row schema matches step3.run(): {id, chapter, page, sent_idx, sinonom,
    am_han_viet, box_ids, is_dbl}. ``box_ids`` are GLOBAL box ``id`` strings; ``page``
    is the sentence's first box's page. Rows come out in reading order (main + 雙行
    interleaved by their first box), sent_idx numbered per page.
    """
    chapters: dict[int, list[dict]] = defaultdict(list)
    for b in han_boxes:
        chapters[b["chapter"]].append(b)

    def _emit(chapter: int, seg: str, boxes: list[dict], is_dbl: bool, pos: int) -> dict:
        # unique box ids in first-appearance (reading) order
        seen, ids = set(), []
        for b in boxes:
            if b["id"] not in seen:
                seen.add(b["id"]); ids.append(b["id"])
        return {"_pos": pos, "chapter": chapter, "page": boxes[0]["page"],
                "sinonom": seg, "am_han_viet": transliterate(seg, hanviet),
                "box_ids": ids, "is_dbl": is_dbl}

    out: list[dict] = []
    for chapter in sorted(chapters):
        cols = sorted(chapters[chapter], key=lambda b: (b["page"], b["box_idx"]))
        pos_of = {b["id"]: i for i, b in enumerate(cols)}   # global reading-order index

        main = [b for b in cols if not b.get("is_dbl")]
        chars: list[str] = []
        owner: list[dict] = []
        for b in main:
            for ch in b["sinonom"]:
                chars.append(ch)
                owner.append(b)

        rows: list[dict] = []
        if chars:
            marks = labels_fn("".join(chars))
            if len(marks) != len(chars):                    # defensive: labels_fn must align
                marks = (list(marks) + [""] * len(chars))[:len(chars)]
            for (a, e) in _split_stream(chars, marks, sent_marks):
                seg = "".join(chars[a:e])
                if seg:
                    rows.append(_emit(chapter, seg, owner[a:e], False, pos_of[owner[a]["id"]]))

        for b in cols:
            if b.get("is_dbl") and b["sinonom"]:
                rows.append(_emit(chapter, b["sinonom"], [b], True, pos_of[b["id"]]))

        rows.sort(key=lambda r: r.pop("_pos"))              # reading order (main + 雙行)
        per_page: dict[int, int] = defaultdict(int)
        for r in rows:
            per_page[r["page"]] += 1
            r["sent_idx"] = per_page[r["page"]]
            r["id"] = make_id(chapter, r["page"], r["sent_idx"])
            out.append(r)
    return out

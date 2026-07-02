"""Manual corrections to the v4 Hán OCR run (det box_thresh 0.4, 2026-07-02).

Three pages carry text that no detector/consensus configuration recovers —
columns swallowed by adjacent 雙行 zones, plus recognition errors on the same
pages, all verified char-by-char against the page scans:

  vol3 p314  cols 5-6 sat next to an interlinear note: 鎮/禱 dropped inside
             col 5, the single-char column 聞 never detected; plus rec errors
             on cols 1-3, 6-7 (雨→而, 閒→閦, 遣→逻, 穀 dropped, 旬→甸, 禱→祷).
  vol1 p390  main col: 尊 dropped, 願/準 misread; its interlinear note
             (向例額定尊室入監讀書以六十人為限) detected as the 2-char
             fragment 人名; rec errors on cols 5-8.
  vol5 p288  the main-size column 四曰錯誤處分 and the 隨職 head of the next
             column were swallowed by the garbled 雙行 zone.

Applies in place to an extracted corpus dir (vol{N}/han_boxes.jsonl +
han_sentences.jsonl). Inserted boxes shift later box ids on their page by +1;
all box_ids references in sentences are remapped. am_han_viet is regenerated
for every record whose text changed.

Run:  python tools/manual_fixes_v4.py <corpus_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.step3_sinonom import load_hanviet, transliterate   # noqa: E402

# ---------------------------------------------------------------------------
# Edit tables. BOX_TEXT / SENT_TEXT: id -> (old_substring, new_substring);
# old must occur exactly once so a re-run or a drifted input fails loudly.
# ---------------------------------------------------------------------------
BOX_TEXT = {
    "vol3": {
        "HVH_001.001.314.01": ("京師得而", "京師得雨"),
        "HVH_001.001.314.02": ("審外閦何如可逻人", "審外閒何如可遣人"),
        "HVH_001.001.314.03": ("常年熟察", "常年穀熟察"),
        "HVH_001.001.314.05": ("久旱臣不報及得雨", "久旱鎮臣不報及禱得雨"),
        "HVH_001.001.314.06": ("職在甸宣", "職在旬宣"),
        "HVH_001.001.314.07": ("竭誠祈祷", "竭誠祈禱"),
    },
    "vol1": {
        "HVH_001.001.390.01": ("訓诲爾等", "訓誨爾等"),
        "HVH_001.001.390.03": ("命室子弟有愿從學者皆准入監向",
                               "命尊室子弟有願從學者皆準入監"),
        "HVH_001.001.390.04": ("人名", "向例額定尊室入監讀書以六十人為限"),
        "HVH_001.001.390.05": ("天潢派俊秀諒不乏人若柴拘定额",
                               "天潢派系俊秀諒不乏人若槩拘定額"),
        "HVH_001.001.390.06": ("朋遊擇各系", "朋遴擇各系"),
        "HVH_001.001.390.07": ("如有願敏好學者不拘多寡核定題明",
                               "如有頴敏好學者不拘多寡核寔題明"),
    },
    "vol5": {
        # id BEFORE renumber (becomes .07 after the p288 insert below)
        "HVH_001.001.288.06": ("名酌分四等", "隨職名酌分四等"),
    },
}

# the p390 note column is interlinear small text -> tag dbl (excluded from NB3)
BOX_SET_DBL = {"vol1": ["HVH_001.001.390.04"]}

# New boxes: (vol, page, insert_before_ss, record-without-id). Later ids on the
# page shift +1. `insert_before_ss` = current ss the new box displaces.
NEW_BOXES = [
    # 聞 column p314: production det misses it; bbox/conf measured by a
    # low-threshold detector pass that does catch it.
    ("vol3", 314, 6, dict(chapter=1, page=314, bbox=[552, 507, 666, 630],
                          sinonom="聞", conf=0.988, is_dbl=False)),
    # 四曰錯誤處分 p288: never detected; bbox estimated from the scan.
    ("vol5", 288, 5, dict(chapter=1, page=288, bbox=[868, 570, 955, 1610],
                          sinonom="四曰錯誤處分", conf=1.0, is_dbl=False)),
]

SENT_TEXT = {
    "vol3": {
        "HVH_001.001.314.01": ("得而朕深", "得雨朕深"),
        "HVH_001.001.314.02": ("外閦", "外閒"),
        "HVH_001.001.314.03": ("可逻人", "可遣人"),
        "HVH_001.001.314.04": ("常年熟察", "常年穀熟察"),
        "HVH_001.001.314.05": ("久旱臣不報", "久旱鎮臣不報"),
    },
    "vol1": {
        "HVH_001.001.389.14": ("訓诲爾等", "訓誨爾等"),
        "HVH_001.001.390.03": ("命室子弟有愿從學者皆准入監向人名",
                               "命尊室子弟有願從學者皆準入監"),
        "HVH_001.001.390.04": ("天潢派俊秀", "天潢派系俊秀"),
        "HVH_001.001.390.06": ("若柴拘定额", "若槩拘定額"),
        "HVH_001.001.390.07": ("朋遊擇各系如有願敏", "朋遴擇各系如有頴敏"),
        "HVH_001.001.390.08": ("核定題明", "核寔題明"),
    },
    "vol5": {
        "HVH_001.001.287.04": ("吏長京外諸衙門各名酌",
                               "吏長四曰錯誤處分京外諸衙門各隨職名酌"),
    },
}

# vol3 p314: old sentence .06 fused two real sentences because 聞 was missing.
# Split it; ids after the split point shift +1 (handled in rebuild below).
SPLIT_SENT = {
    "vol3": {
        "HVH_001.001.314.06": [
            # (new sinonom, box_ids AFTER box renumber)
            ("及禱得雨始以聞",
             ["HVH_001.001.314.05", "HVH_001.001.314.06"]),
            ("帝譴之曰爾等職在旬宣一遇亢旱當卽飛章入奏竭誠祈禱庶幾早沐甘霖旱苗可救",
             ["HVH_001.001.314.07", "HVH_001.001.314.08"]),
        ],
    },
}

# vol1 p390: the note column becomes its own dbl sentence after .03.
INSERT_SENT = {
    "vol1": [
        ("HVH_001.001.390.03",       # insert after this sentence id
         dict(chapter=1, page=390,
              sinonom="向例額定尊室入監讀書以六十人為限",
              box_ids=["HVH_001.001.390.04"], is_dbl=True)),
    ],
}

# sentence id -> box id (post-renumber) to add to its box_ids: the p287
# cross-page sentence gains the inserted 四曰錯誤處分 box.
SENT_ADD_BOX = {"vol5": {"HVH_001.001.287.04": "HVH_001.001.288.05"}}


def _sent_id(page_id_prefix: str, page: int, idx: int) -> str:
    return f"{page_id_prefix}.{page:03d}.{idx:02d}"


def apply(root: Path) -> None:
    hanviet = load_hanviet()

    for vol in ["vol1", "vol3", "vol5"]:
        vdir = root / vol
        boxes = [json.loads(l) for l in open(vdir / "han_boxes.jsonl") if l.strip()]
        sents = [json.loads(l) for l in open(vdir / "han_sentences.jsonl") if l.strip()]

        # -- 1. box text edits (ids are pre-renumber; must run before inserts,
        #       which temporarily duplicate an id on the page) ----------------
        for bid, (old, newt) in BOX_TEXT.get(vol, {}).items():
            b = next(x for x in boxes if x["id"] == bid)
            assert b["sinonom"].count(old) == 1, (vol, bid, old, b["sinonom"])
            b["sinonom"] = b["sinonom"].replace(old, newt)
            b["am_han_viet"] = transliterate(b["sinonom"], hanviet)
        for bid in BOX_SET_DBL.get(vol, []):
            next(x for x in boxes if x["id"] == bid)["is_dbl"] = True

        # -- 2. insert new boxes: shift later ids on the page +1 (applied
        #       immediately, BEFORE the insert, so the new box's own id can
        #       never collide with a remap key), then slot the new box in ----
        remap: dict[str, str] = {}
        for nvol, page, before_ss, rec in NEW_BOXES:
            if nvol != vol:
                continue
            prefix = boxes[0]["id"].rsplit(".", 2)[0]          # HVH_001.ccc
            for b in boxes:
                if b["page"] != page:
                    continue
                ss = int(b["id"].rsplit(".", 1)[1])
                if ss >= before_ss:
                    remap[b["id"]] = f"{prefix}.{page:03d}.{ss + 1:02d}"
                    b["id"] = remap[b["id"]]
                    b["box_idx"] = ss + 1
            new = dict(rec, id=f"{prefix}.{page:03d}.{before_ss:02d}",
                       box_idx=before_ss,
                       am_han_viet=transliterate(rec["sinonom"], hanviet))
            pos = next(i for i, b in enumerate(boxes)
                       if b["page"] == page and int(b["id"].rsplit(".", 1)[1]) > before_ss)
            boxes.insert(pos, new)

        # -- 4. sentence edits -------------------------------------------------
        for s in sents:
            s["box_ids"] = [remap.get(i, i) for i in s["box_ids"]]
        for sid, add in SENT_ADD_BOX.get(vol, {}).items():
            s = next(x for x in sents if x["id"] == sid)
            s["box_ids"] = sorted(set(s["box_ids"]) | {add})
        for sid, (old, newt) in SENT_TEXT.get(vol, {}).items():
            s = next(x for x in sents if x["id"] == sid)
            assert s["sinonom"].count(old) == 1, (vol, sid, old, s["sinonom"])
            s["sinonom"] = s["sinonom"].replace(old, newt)
            s["am_han_viet"] = transliterate(s["sinonom"], hanviet)

        # splits / inserts shift later sentence ids on the same page by +1
        def renumber_after(page: int, from_idx: int) -> None:
            prefix = sents[0]["id"].rsplit(".", 2)[0]
            for s in sents:
                if s["page"] == page and s["sent_idx"] >= from_idx:
                    s["sent_idx"] += 1
                    s["id"] = _sent_id(prefix, page, s["sent_idx"])

        for sid, parts in SPLIT_SENT.get(vol, {}).items():
            i = next(i for i, s in enumerate(sents) if s["id"] == sid)
            base = sents[i]
            renumber_after(base["page"], base["sent_idx"] + 1)
            prefix = base["id"].rsplit(".", 2)[0]
            repl = []
            for k, (txt, bids) in enumerate(parts):
                repl.append(dict(base, sinonom=txt, box_ids=bids,
                                 am_han_viet=transliterate(txt, hanviet),
                                 sent_idx=base["sent_idx"] + k,
                                 id=_sent_id(prefix, base["page"], base["sent_idx"] + k)))
            sents[i:i + 1] = repl

        for after_id, rec in INSERT_SENT.get(vol, []):
            i = next(i for i, s in enumerate(sents) if s["id"] == after_id)
            base = sents[i]
            renumber_after(base["page"], base["sent_idx"] + 1)
            prefix = base["id"].rsplit(".", 2)[0]
            new = dict(rec, sent_idx=base["sent_idx"] + 1,
                       am_han_viet=transliterate(rec["sinonom"], hanviet),
                       id=_sent_id(prefix, base["page"], base["sent_idx"] + 1))
            sents.insert(i + 1, new)

        # -- 5. write back -----------------------------------------------------
        for name, rows in [("han_boxes.jsonl", boxes), ("han_sentences.jsonl", sents)]:
            with open(vdir / name, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{vol}] patched: {len(boxes)} boxes, {len(sents)} sentences")


if __name__ == "__main__":
    apply(Path(sys.argv[1]))

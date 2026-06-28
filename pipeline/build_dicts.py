"""Build the lookup resources used by steps 3 and 4.

Sources (all public / authoritative):
  * Unicode Unihan database — `kVietnamese` field gives the âm Hán-Việt reading
    of a CJK character; variant fields give visually/semantically related chars.
    Official but sparse (~8k chars) and occasionally lists a Nôm reading.
  * KanjiDictVN (trungnt2910/KanjiDictVN, EDRDG/KANJIDIC2, CC BY-SA) — curated
    âm Hán-Việt for ~10k chars; fills the chars Unihan kVietnamese misses.
  * hanviet_overrides.csv / hanviet_supplement.csv — curated disambiguation +
    last-resort fills (a char often has both a Hán-Việt and a Nôm reading; no
    source can pick the right primary for THIS corpus automatically).
  * Course sample dictionaries (QuocNgu_SinoNom.dic / SinoNom_Similar.dic) —
    used as a format reference and merged in (they take priority where present).
  * Viet74K Vietnamese wordlist (duyet/vietnamese-wordlist) — used by step 1/2.

Outputs (written to assets/dicts/):
  * hanviet.csv                 sinonom_char, am_han_viet, all_readings
  * QuocNgu_SinoNom.dic  (S2)   <quoc_ngu>:<sn1> <sn2> ...     (rebuilt + merged)
  * SinoNom_Similar.dic  (S1)   <sn>:<sim1> <sim2> ...         (rebuilt + merged)

Run:  python -m pipeline.build_dicts
"""
from __future__ import annotations

import csv
import zipfile
from collections import defaultdict
from pathlib import Path

from . import config
from .common import get_logger, is_cjk

log = get_logger("build_dicts")

UNIHAN_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"


# --------------------------------------------------------------------------- #
# Unihan parsing
# --------------------------------------------------------------------------- #
def ensure_unihan() -> None:
    """Make sure Unihan_Readings.txt / Unihan_Variants.txt exist (download if not)."""
    need = [config.UNIHAN_READINGS, config.UNIHAN_VARIANTS]
    if all(p.exists() for p in need):
        return
    import urllib.request

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = config.RAW_DIR / "Unihan.zip"
    if not zpath.exists():
        log.info("downloading Unihan database ...")
        urllib.request.urlretrieve(UNIHAN_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for member in ("Unihan_Readings.txt", "Unihan_Variants.txt"):
            z.extract(member, config.RAW_DIR)
    log.info("Unihan extracted to %s", config.RAW_DIR)


def _cp_to_char(token: str) -> str:
    """'U+346B' -> the character."""
    return chr(int(token[2:], 16))


# Normalise Vietnamese diacritic placement to the traditional Hán-Việt
# convention (diacritic on the first vowel: hòa, thúy, thụy) so readings from
# different sources match in S2 lookup. KanjiDictVN uses the new style (hoà,
# thuý, thuỵ); Unihan / curated data use the old style.
_TONE_NORM = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}


def norm_tone(syllable: str) -> str:
    s = syllable.strip().lower()
    for new, old in _TONE_NORM.items():
        if new in s:
            s = s.replace(new, old)
    return s


def ensure_kanjidictvn() -> None:
    """Download + cache the KanjiDictVN reading banks (if not already present)."""
    if all(p.exists() for p in config.KANJIDICTVN_FILES):
        return
    import urllib.request

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url, path in zip(config.KANJIDICTVN_URLS, config.KANJIDICTVN_FILES):
        if path.exists():
            continue
        log.info("downloading KanjiDictVN %s ...", path.name)
        urllib.request.urlretrieve(url, path)


def parse_kanjidictvn() -> dict[str, list[str]]:
    """char -> ordered list of âm Hán-Việt readings, from KanjiDictVN.

    Yomichan kanji-bank entry = [char, onyomi, kunyomi, tags, [meanings], stats];
    onyomi holds the space-separated âm Hán-Việt. Returns {} if files missing.
    """
    import json as _json

    readings: dict[str, list[str]] = {}
    for path in config.KANJIDICTVN_FILES:
        if not path.exists():
            continue
        for entry in _json.loads(path.read_text(encoding="utf-8")):
            ch = entry[0]
            raw = (entry[1] or entry[2] or "").split()
            rs: list[str] = []
            for r in raw:
                r = norm_tone(r)
                if r and r not in rs:
                    rs.append(r)
            if rs:
                readings.setdefault(ch, rs)
    if readings:
        log.info("KanjiDictVN readings: %d chars", len(readings))
    return readings


def load_overrides() -> dict[str, str]:
    """Manual corrections char -> âm Hán-Việt (assets/dicts/hanviet_overrides.csv)."""
    path = config.DICTS_DIR / "hanviet_overrides.csv"
    overrides: dict[str, str] = {}
    if not path.exists():
        return overrides
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith(("#", "sinonom_char")):
                continue
            if len(row) >= 2 and row[0].strip():
                overrides[row[0].strip()] = row[1].strip()
    if overrides:
        log.info("loaded %d manual hanviet overrides", len(overrides))
    return overrides


def load_supplement() -> dict[str, str]:
    """Curated char -> âm Hán-Việt for chars the Unihan kVietnamese field misses.

    kVietnamese only covers ~8k chars, so common chars like 亦 (diệc), 北 (bắc),
    祖 (tổ) come out blank and step 3 renders them as '?'. This file (highest
    priority) fills that gap; see assets/dicts/hanviet_supplement.csv.
    """
    path = config.HANVIET_SUPPLEMENT
    supp: dict[str, str] = {}
    if not path.exists():
        return supp
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith(("#", "sinonom_char")):
                continue
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                supp.setdefault(row[0].strip(), row[1].strip())
    if supp:
        log.info("loaded %d curated hanviet supplements", len(supp))
    return supp


def invert_dic_readings(s2: dict[str, list[str]]) -> dict[str, list[str]]:
    """Invert a course S2 dict (quoc_ngu -> [sinonom chars]) into char -> [readings].

    Recovers any curated teacher readings that are not in Unihan kVietnamese.
    """
    inv: dict[str, list[str]] = {}
    for qn, chars in s2.items():
        for ch in chars:
            inv.setdefault(ch, [])
            if qn not in inv[ch]:
                inv[ch].append(qn)
    return inv


def parse_readings(overrides: dict[str, str] | None = None) -> dict[str, list[str]]:
    """char -> ordered list of âm Hán-Việt readings (from kVietnamese).

    A manual override is prepended (so it becomes the primary reading) without
    discarding the Unihan readings.
    """
    overrides = overrides or {}
    readings: dict[str, list[str]] = {}
    with config.UNIHAN_READINGS.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\tkVietnamese\t" not in line:
                continue
            cp, _field, value = line.rstrip("\n").split("\t", 2)
            ch = _cp_to_char(cp)
            # values are space separated, lower-cased syllables
            readings[ch] = [norm_tone(v) for v in value.split() if v.strip()]
    for ch, am in overrides.items():
        am = norm_tone(am)
        rest = [r for r in readings.get(ch, []) if r != am]
        readings[ch] = [am] + rest
    log.info("kVietnamese readings: %d chars (%d overridden)", len(readings), len(overrides))
    return readings


def parse_variants() -> dict[str, set[str]]:
    """char -> set of related chars (semantic/Z/shape variants).

    Used as a *fallback* source of visual-similarity candidates (S1) since the
    full course SinoNom_Similar.dic is not publicly available. Variants are a
    reasonable proxy: they cover the most common confusions in OCR.
    """
    fields = ("kSemanticVariant", "kZVariant", "kSimplifiedVariant",
              "kTraditionalVariant", "kSpecializedSemanticVariant")
    variants: dict[str, set[str]] = defaultdict(set)
    with config.UNIHAN_VARIANTS.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            cp, field, value = line.rstrip("\n").split("\t", 2)
            if field not in fields:
                continue
            ch = _cp_to_char(cp)
            for tok in value.split():
                tok = tok.split("<")[0]              # strip "<kHanYu" provenance tags
                if tok.startswith("U+"):
                    variants[ch].add(_cp_to_char(tok))
    log.info("variant candidates: %d chars", len(variants))
    return variants


# --------------------------------------------------------------------------- #
# Sample-dic parsing (course-provided format)
# --------------------------------------------------------------------------- #
def parse_dic(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        out[key.strip()] = [t for t in rest.split() if t.strip()]
    return out


def write_dic(path: Path, mapping: dict[str, list[str]], header: str) -> None:
    lines = [f"# {header}", "# Encoding: UTF-8", "#"]
    for key, vals in mapping.items():
        lines.append(f"{key}:{' '.join(vals)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s (%d entries)", path.name, len(mapping))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> None:
    config.DICTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_unihan()
    ensure_kanjidictvn()

    overrides = load_overrides()
    supplement = load_supplement()
    readings = parse_readings(overrides)        # Unihan kVietnamese, override-corrected
    kanji = parse_kanjidictvn()                 # KanjiDictVN (EDRDG) — fills the gaps
    variants = parse_variants()

    # Keep the original samples so we never lose curated entries.
    sample_s2 = parse_dic(config.QUOCNGU_SINONOM_DIC)
    sample_s1 = parse_dic(config.SINONOM_SIMILAR_DIC)
    teacher_inv = invert_dic_readings(sample_s2)

    def variant_reading(ch: str) -> str | None:
        """Reading borrowed from a variant char (handles simplified OCR output,
        e.g. 兴 -> 興 -> 'hưng', 学 -> 學 -> 'học')."""
        for v in sorted(variants.get(ch, set())):
            for src in (readings, kanji):
                if src.get(v):
                    return src[v][0]
        return None

    # --- hanviet.csv: sinonom char -> âm Hán Việt ------------------------- #
    # Priority for the PRIMARY reading:
    #   override > Unihan kVietnamese > KanjiDictVN > supplement > teacher > variant
    # Unihan is the official/citable base; KanjiDictVN (EDRDG) fills the chars it
    # misses (kVietnamese covers only ~8k); overrides/supplement are the curated
    # disambiguation layer (a char often has both a Hán-Việt and a Nôm reading and
    # no source can pick the right primary for THIS corpus automatically).
    all_chars = {c for c in set(readings) | set(kanji) | set(supplement)
                 | set(teacher_inv) | set(variants) if is_cjk(c)}
    rows = []
    n_kviet = n_kanji = n_supp = n_teacher = n_variant = 0
    for ch in sorted(all_chars):
        base = readings.get(ch, [])
        kjv = kanji.get(ch, [])
        if base:
            primary = base[0]; n_kviet += 1
        elif kjv:
            primary = kjv[0]; n_kanji += 1
        elif ch in supplement:
            primary = norm_tone(supplement[ch]); n_supp += 1
        elif ch in teacher_inv:
            primary = teacher_inv[ch][0]; n_teacher += 1
        else:
            primary = variant_reading(ch)
            if not primary:
                continue
            n_variant += 1
        all_rd: list[str] = []
        for r in [primary, *base, *kjv, *teacher_inv.get(ch, [])]:
            if r and r not in all_rd:
                all_rd.append(r)
        rows.append((ch, primary, " ".join(all_rd)))
    with config.HANVIET_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sinonom_char", "am_han_viet", "all_readings"])
        w.writerows(rows)
    log.info("wrote %s (%d chars: %d kVietnamese, %d KanjiDictVN, %d supplement, "
             "%d teacher, %d variant)", config.HANVIET_CSV.name, len(rows),
             n_kviet, n_kanji, n_supp, n_teacher, n_variant)

    # --- S2: Quốc ngữ -> {SinoNom chars} (invert readings) ---------------- #
    # Invert the SAME enriched char->readings we wrote to hanviet.csv (supplement
    # + variant-fallback + teacher included), so step 4's S1∩S2 char validation
    # recognises those chars (otherwise they'd read fine but be flagged RED).
    s2: dict[str, set[str]] = defaultdict(set)
    for word, chars in sample_s2.items():
        s2[word].update(chars)
    for ch, _primary, all_rd in rows:
        for syll in all_rd.split():
            s2[syll].add(ch)
    s2_out = {k: sorted(v) for k, v in sorted(s2.items())}
    write_dic(config.QUOCNGU_SINONOM_DIC, s2_out,
              "S2: Quoc Ngu -> SinoNom chars (Unihan kVietnamese, inverted + sample merge)")

    # --- S1: SinoNom char -> [visually similar] --------------------------- #
    s1: dict[str, list[str]] = {}
    keys = set(sample_s1) | set(variants)
    for ch in sorted(keys):
        ordered: list[str] = []
        for c in sample_s1.get(ch, []):          # curated first (similarity order)
            if c not in ordered:
                ordered.append(c)
        for c in sorted(variants.get(ch, set())):
            if c not in ordered:
                ordered.append(c)
        if ch not in ordered:
            ordered.append(ch)                    # a char is similar to itself
        s1[ch] = ordered
    write_dic(config.SINONOM_SIMILAR_DIC, s1,
              "S1: SinoNom -> visually similar chars (sample + Unihan variants)")

    log.info("done. S1=%d  S2=%d  hanviet=%d", len(s1), len(s2_out), len(rows))


if __name__ == "__main__":
    build()

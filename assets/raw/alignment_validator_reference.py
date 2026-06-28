"""alignment_validator.py
=======================
Minimum Edit Distance (Levenshtein) alignment validator for SinoNom OCR.

This module implements the character-alignment algorithm for SinoNom OCR validation
using visual similarity and transliteration mappings.

Algorithm Summary
-----------------
For each OCR-recognised SinoNom character ``sn`` paired with a Quoc Ngu
(romanised Vietnamese) word ``qn``, determine correctness by set intersection:

  S1 = { chars visually similar to sn  }  (from SinoNom_Similar.dic)
  S2 = { chars valid translations of qn }  (from QuocNgu_SinoNom.dic)

Decision rules:
  - If sn ∈ S2              → BLACK   (correct OCR, matches QN directly)
  - If sn ∉ S2:
      Compute G = S1 ∩ S2
      ├─ len(G) == 1 → GREEN  (unique correction found — take it)
      ├─ len(G) >  1 → GREEN  (take highest-ranked char in S1 order)
      └─ len(G) == 0 → RED    (OCR failure — cannot correct)

Status colours map to standardised :class:`AlignmentStatus` enum values.

Additionally, this module implements the full Levenshtein sequence alignment
so that an entire OCR character sequence can be aligned with a QN token sequence,
producing an edit-script with per-character status annotations.

Author: SinoNom OCR Project Contributors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from rapidfuzz.distance import Levenshtein

logger = logging.getLogger("alignment_validator")


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class AlignmentStatus(str, Enum):
    """Colour-coded OCR alignment status for a single SinoNom character.

    Values mirror standard color conventions:
    - BLACK: OCR is correct and matches the expected QN character.
    - GREEN: OCR was incorrect but corrected via S1∩S2 lookup.
    - RED:   OCR failure — neither direct match nor correction found.
    """

    BLACK = "BLACK"  # Correct: sn ∈ S2
    GREEN = "GREEN"  # Corrected: unique or best match found in S1 ∩ S2
    RED = "RED"  # Failure: S1 ∩ S2 is empty


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CharAlignmentResult:
    """Alignment result for one (sn, qn) character pair.

    Attributes:
        sn:               OCR-recognised SinoNom character.
        qn:               Corresponding Quoc Ngu syllable/word.
        status:           :class:`AlignmentStatus` verdict.
        corrected_char:   The corrected SinoNom char (GREEN status only).
        s1_candidates:    Set S1 — visually similar chars to ``sn``.
        s2_candidates:    Set S2 — valid QN translations for ``qn``.
        intersection:     G = S1 ∩ S2.
    """

    sn: str
    qn: str
    status: AlignmentStatus
    corrected_char: str | None = None
    s1_candidates: set[str] = field(default_factory=set)
    s2_candidates: set[str] = field(default_factory=set)
    intersection: set[str] = field(default_factory=set)

    def __str__(self) -> str:
        correction = f" → corrected={self.corrected_char!r}" if self.corrected_char else ""
        return f"CharAlign(sn={self.sn!r}, qn={self.qn!r}, status={self.status.value}{correction})"


@dataclass
class SequenceAlignmentResult:
    """Alignment result for a full box (sequence of SinoNom chars vs QN words).

    Attributes:
        ocr_sequence:     List of raw OCR characters.
        qn_sequence:      List of Quoc Ngu syllables/words.
        char_results:     Per-character :class:`CharAlignmentResult` list.
        edit_distance:    Levenshtein distance between the two sequences
                          (using character identity before correction).
        normalised_score: ``1 - edit_distance / max(len(sn_seq), len(qn_seq))``
    """

    ocr_sequence: list[str]
    qn_sequence: list[str]
    char_results: list[CharAlignmentResult] = field(default_factory=list)
    edit_distance: int = 0
    normalised_score: float = 0.0

    @property
    def accuracy(self) -> float:
        """Fraction of characters with BLACK or GREEN status (correctable)."""
        if not self.char_results:
            return 0.0
        good = sum(
            1
            for r in self.char_results
            if r.status in (AlignmentStatus.BLACK, AlignmentStatus.GREEN)
        )
        return good / len(self.char_results)


# ---------------------------------------------------------------------------
# Mapping dictionaries
# ---------------------------------------------------------------------------

# S1: SinoNom_Similar.dic
# Maps each SinoNom character to a list of visually similar characters.
# The ORDER within each list is significant — used for tie-breaking (leftmost wins).
# In production, this is loaded from the actual .dic file.
# Here we pre-seed with canonical examples plus
# a comprehensive representative subset.
SINONOM_SIMILAR_S1: dict[str, list[str]] = {
    # Canonical example (Truyện Kiều, first character "百"):
    "百": ["𤾓", "榥", "釈", "椩", "棅", "稂", "百", "佒"],
    # Common Sino-Nom character confusables
    "人": ["入", "八", "人"],
    "大": ["太", "犬", "大"],
    "日": ["曰", "目", "日"],
    "月": ["用", "勿", "月"],
    "山": ["屮", "凸", "山"],
    "水": ["氺", "氵", "水"],
    "火": ["灬", "火"],
    "木": ["本", "末", "木"],
    "金": ["釒", "金"],
    "土": ["士", "土"],
    "王": ["玨", "壬", "王"],
    "帝": ["啻", "蒂", "帝"],
    "聖": ["𦣣", "聖"],
    "德": ["惪", "德"],
    "新": ["薪", "新"],
    "臨": ["𡦫", "臨"],
    "朝": ["嘲", "朝"],
    "異": ["巽", "異"],
    "盛": ["𦚟", "盛"],
    "年": ["秊", "年"],
    "身": ["躳", "身"],
    "後": ["後", "後"],
    "名": ["名", "各"],
    # Additional common characters for pipeline robustness
    "心": ["忄", "心"],
    "手": ["扌", "手"],
    "口": ["囗", "口"],
    "目": ["罒", "目"],
    "耳": ["耳"],
    "足": ["⻊", "足"],
    "言": ["訁", "言"],
    "走": ["赱", "走"],
    "力": ["刀", "力"],
    "刀": ["力", "刀"],
    "弓": ["弔", "弓"],
}

# S2: QuocNgu_SinoNom.dic
# Maps each Quoc Ngu syllable/word to the set of SinoNom chars that can
# represent it (transcription/translation candidates).
# In production, this is a large dictionary loaded from file.
QUOCNGU_SINONOM_S2: dict[str, set[str]] = {
    # Canonical example: "trăm" maps to {百, 𬃴, …}
    "trăm": {"百", "𬃴", "𤾓"},
    "năm": {"年", "南", "𢆥"},
    "thân": {"身", "親", "𨉟"},
    "sau": {"後", "𠫾"},
    "danh": {"名", "𠊛"},
    # Common Vietnamese syllables → SinoNom mappings
    "vua": {"王", "𤤰"},
    "hoàng": {"皇", "黃"},
    "đế": {"帝"},
    "thánh": {"聖"},
    "đức": {"德"},
    "mới": {"新"},
    "lâm": {"臨"},
    "triều": {"朝"},
    "ngày": {"日"},
    "tháng": {"月"},
    "núi": {"山"},
    "nước": {"水", "國"},
    "lửa": {"火"},
    "cây": {"木"},
    "vàng": {"金"},
    "đất": {"土"},
    "người": {"人", "𠊛"},
    "lòng": {"心", "𢚸"},
    "tay": {"手"},
    "miệng": {"口"},
    "mắt": {"目"},
    "tai": {"耳"},
    "chân": {"足"},
    "lời": {"言"},
    "chạy": {"走"},
    "sức": {"力"},
    "khác": {"異"},
    "thịnh": {"盛"},
    "lạ": {"異"},
}


# ---------------------------------------------------------------------------
# Core validator class
# ---------------------------------------------------------------------------


class SinoNomAlignmentValidator:
    """Validates OCR character correctness using the S1∩S2 algorithm.

    Args:
        s1_dict: Mapping from SinoNom character to list of visually similar chars.
                 Defaults to the built-in :data:`SINONOM_SIMILAR_S1`.
        s2_dict: Mapping from Quoc Ngu syllable to set of SinoNom translations.
                 Defaults to the built-in :data:`QUOCNGU_SINONOM_S2`.
    """

    def __init__(
        self,
        s1_dict: dict[str, list[str]] | None = None,
        s2_dict: dict[str, set[str]] | None = None,
        hanviet_path: str | None = None,
    ) -> None:
        self._s1: dict[str, list[str]] = s1_dict or SINONOM_SIMILAR_S1
        self._s2: dict[str, set[str]] = s2_dict or QUOCNGU_SINONOM_S2

        # Load and merge hanviet.csv if provided and exists
        if hanviet_path:
            try:
                hv_s2 = load_hanviet_csv(hanviet_path)
                for qn, chars in hv_s2.items():
                    if qn not in self._s2:
                        self._s2[qn] = set()
                    self._s2[qn].update(chars)
            except Exception as e:
                logger.warning("Failed to load hanviet_path %s: %s", hanviet_path, e)

    # ------------------------------------------------------------------
    def validate_pair(self, sn: str, qn: str) -> CharAlignmentResult:
        """Validate a single (OCR char, Quoc Ngu word) pair.

        Implements the core alignment algorithm:

        ::

            if sn in S2(qn):
                → BLACK (correct)
            else:
                G = S1(sn) ∩ S2(qn)
                if len(G) >= 1:
                    → GREEN (corrected to best G candidate)
                else:
                    → RED (OCR failure)

        Args:
            sn: The OCR-recognised SinoNom character.
            qn: The aligned Quoc Ngu syllable/word.

        Returns:
            A :class:`CharAlignmentResult` with full diagnostic information.
        """
        # Retrieve S2: translations valid for this qn syllable
        s2_candidates: set[str] = self._s2.get(qn.lower().strip(), set())

        # Retrieve S1: visually similar characters to sn
        s1_candidates_ordered: list[str] = self._s1.get(sn, [sn])
        s1_candidates: set[str] = set(s1_candidates_ordered)

        # Decision rule 1: sn ∈ S2 → BLACK (correct)
        if sn in s2_candidates:
            logger.debug("PAIR (%r, %r) → BLACK (sn ∈ S2)", sn, qn)
            return CharAlignmentResult(
                sn=sn,
                qn=qn,
                status=AlignmentStatus.BLACK,
                corrected_char=None,
                s1_candidates=s1_candidates,
                s2_candidates=s2_candidates,
                intersection=set(),
            )

        # Decision rule 2: compute intersection G = S1 ∩ S2
        intersection: set[str] = s1_candidates & s2_candidates

        if len(intersection) >= 1:
            # GREEN: find best candidate (leftmost in S1's ordered list)
            corrected = self._pick_best_from_intersection(
                s1_ordered=s1_candidates_ordered,
                intersection=intersection,
            )
            logger.debug(
                "PAIR (%r, %r) → GREEN (|G|=%d, corrected→%r)",
                sn,
                qn,
                len(intersection),
                corrected,
            )
            return CharAlignmentResult(
                sn=sn,
                qn=qn,
                status=AlignmentStatus.GREEN,
                corrected_char=corrected,
                s1_candidates=s1_candidates,
                s2_candidates=s2_candidates,
                intersection=intersection,
            )

        # Decision rule 3: G = ∅ → RED
        logger.debug("PAIR (%r, %r) → RED (G = ∅)", sn, qn)
        return CharAlignmentResult(
            sn=sn,
            qn=qn,
            status=AlignmentStatus.RED,
            corrected_char=None,
            s1_candidates=s1_candidates,
            s2_candidates=s2_candidates,
            intersection=set(),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _pick_best_from_intersection(
        s1_ordered: list[str],
        intersection: set[str],
    ) -> str:
        """Select the highest-ranked intersection candidate.

        Tie-breaking rule: when |G| > 1, pick the character
        that appears leftmost (earliest index) in the S1 ordered list,
        as S1 is assumed to be ordered by visual similarity descending.

        Args:
            s1_ordered:  S1 as an ordered list (most similar first).
            intersection: Set G = S1 ∩ S2.

        Returns:
            The best candidate character string.
        """
        for char in s1_ordered:
            if char in intersection:
                return char
        # Fallback: return any element (should not normally reach here)
        return next(iter(intersection))

    # ------------------------------------------------------------------
    def validate_sequence(
        self,
        sn_sequence: list[str],
        qn_sequence: list[str],
    ) -> SequenceAlignmentResult:
        """Validate and align a sequence of OCR characters against QN words.

        Uses Levenshtein edit-script alignment to pair up SinoNom characters
        with their corresponding Quoc Ngu syllables, then validates each pair.

        Args:
            sn_sequence: Ordered list of OCR-recognised SinoNom characters.
            qn_sequence: Ordered list of Quoc Ngu syllables (same box).

        Returns:
            A :class:`SequenceAlignmentResult` with per-character verdicts.
        """
        # Compute Levenshtein distance for scoring
        edit_dist = Levenshtein.distance(sn_sequence, qn_sequence)
        max_len = max(len(sn_sequence), len(qn_sequence), 1)
        norm_score = 1.0 - edit_dist / max_len

        # Align the two sequences using the edit script
        aligned_pairs = self._align_with_edit_script(sn_sequence, qn_sequence)

        char_results: list[CharAlignmentResult] = []
        for sn, qn in aligned_pairs:
            if sn == "" or qn == "":
                # Insertion/deletion — mark as RED (unmatched)
                char_results.append(
                    CharAlignmentResult(
                        sn=sn or "∅",
                        qn=qn or "∅",
                        status=AlignmentStatus.RED,
                    )
                )
            else:
                char_results.append(self.validate_pair(sn, qn))

        return SequenceAlignmentResult(
            ocr_sequence=sn_sequence,
            qn_sequence=qn_sequence,
            char_results=char_results,
            edit_distance=edit_dist,
            normalised_score=norm_score,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _align_with_edit_script(
        sn_seq: list[str],
        qn_seq: list[str],
    ) -> list[tuple[str, str]]:
        """Produce a character-level alignment using Needleman-Wunsch DP.

        Computes the classic edit-distance alignment matrix and traces back
        the optimal alignment, yielding (sn_char, qn_char) pairs where
        gaps are represented by empty strings.

        Args:
            sn_seq: Source (OCR) character list.
            qn_seq: Target (QN) character list.

        Returns:
            List of ``(sn_char, qn_char)`` alignment pairs.
        """
        m, n = len(sn_seq), len(qn_seq)

        # DP table: dp[i][j] = min edits to align sn_seq[:i] with qn_seq[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if sn_seq[i - 1] == qn_seq[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + cost,  # Substitution / match
                )

        # Traceback
        pairs: list[tuple[str, str]] = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                cost = 0 if sn_seq[i - 1] == qn_seq[j - 1] else 1
                if dp[i][j] == dp[i - 1][j - 1] + cost:
                    pairs.append((sn_seq[i - 1], qn_seq[j - 1]))
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
                pairs.append((sn_seq[i - 1], ""))  # Deletion
                i -= 1
            else:
                pairs.append(("", qn_seq[j - 1]))  # Insertion
                j -= 1

        pairs.reverse()
        return pairs


# ---------------------------------------------------------------------------
# Dictionary loading utilities
# ---------------------------------------------------------------------------


def load_s1_from_file(filepath: str) -> dict[str, list[str]]:
    """Load the SinoNom_Similar.dic dictionary from a text file.

    Expected file format (one entry per line):
      ``<char>:<similar1> <similar2> <similar3> …``

    Args:
        filepath: Path to the SinoNom_Similar.dic file.

    Returns:
        Dictionary mapping each character to its ordered similarity list.

    Raises:
        FileNotFoundError: If the dictionary file does not exist.
    """
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"S1 dictionary file not found: {filepath}")

    result: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                logger.warning("S1 dict line %d malformed (no ':'): %r", line_num, line)
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            similars = [c.strip() for c in value.split() if c.strip()]
            if key:
                result[key] = similars

    logger.info("Loaded S1 dictionary: %d entries from %s", len(result), filepath)
    return result


def load_s2_from_file(filepath: str) -> dict[str, set[str]]:
    """Load the QuocNgu_SinoNom.dic dictionary from a text file.

    Expected file format (one entry per line):
      ``<quoc_ngu_word>:<sinonom1> <sinonom2> …``

    Args:
        filepath: Path to the QuocNgu_SinoNom.dic file.

    Returns:
        Dictionary mapping Quoc Ngu words to their SinoNom character sets.

    Raises:
        FileNotFoundError: If the dictionary file does not exist.
    """
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"S2 dictionary file not found: {filepath}")

    result: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                logger.warning("S2 dict line %d malformed (no ':'): %r", line_num, line)
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            sinonom_chars = {c.strip() for c in value.split() if c.strip()}
            if key:
                result[key] = sinonom_chars

    logger.info("Loaded S2 dictionary: %d entries from %s", len(result), filepath)
    return result


def load_hanviet_csv(filepath: str) -> dict[str, set[str]]:
    """Load S2 mapping from a hanviet.csv file.

    CSV columns: char, hanviet, pinyin
    Format of hanviet column: "['thướng']" or "['phật', 'phất']"

    Args:
        filepath: Path to the hanviet.csv file.

    Returns:
        Dictionary mapping each Quoc Ngu word to its set of SinoNom character candidates.
    """
    import ast
    import csv
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"HanViet CSV file not found: {filepath}")

    result: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)  # Skip header
        except StopIteration:
            return result

        for row in reader:
            if len(row) < 2:
                continue
            char = row[0].strip()
            raw_readings = row[1].strip()
            if not char or not raw_readings:
                continue
            try:
                readings = ast.literal_eval(raw_readings)
            except Exception:
                # Fallback: simple strip
                readings = [r.strip("'\" ") for r in raw_readings.strip("[]").split(",")]

            for r in readings:
                r = r.strip().lower()
                if r:
                    if r not in result:
                        result[r] = set()
                    result[r].add(char)

    logger.info("Loaded HanViet CSV: %d entries from %s", len(result), filepath)
    return result


# ---------------------------------------------------------------------------
# Smoke test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    validator = SinoNomAlignmentValidator()

    print("=" * 65)
    print("Pair-level validation — Canonical example")
    print("=" * 65)
    test_pairs = [
        ("百", "trăm"),  # → BLACK  (百 ∈ S2{"trăm"})
        ("𤾓", "trăm"),  # → GREEN  (visually similar to 百, in S2)
        ("帝", "đế"),  # → BLACK
        ("聖", "thánh"),  # → BLACK
        ("人", "người"),  # → BLACK
        ("X", "trăm"),  # → RED    (unknown char, no S1 or S2 match)
    ]
    for sn, qn in test_pairs:
        result = validator.validate_pair(sn, qn)
        print(f"  {result}")

    print()
    print("=" * 65)
    print("Sequence-level alignment")
    print("=" * 65)
    sn_seq = ["百", "年", "身", "後", "名"]
    qn_seq = ["trăm", "năm", "thân", "sau", "danh"]
    seq_result = validator.validate_sequence(sn_seq, qn_seq)
    print(f"  Edit distance : {seq_result.edit_distance}")
    print(f"  Norm score    : {seq_result.normalised_score:.3f}")
    print(f"  Accuracy      : {seq_result.accuracy:.1%}")
    for cr in seq_result.char_results:
        print(f"  {cr}")

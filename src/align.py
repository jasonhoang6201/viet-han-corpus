"""B4 · Dóng hàng CÂU Hán ↔ Việt (sản phẩm chính).

Lai 2 lớp như hướng dẫn:
  1. Anchor: niên hiệu / số / tên riêng trùng tuyệt đối hai bên -> ghép chắc.
  2. Embedding cosine (LaBSE) + quy hoạch động cho phép m-n.
Trả về list cặp (idx_han_set, idx_viet_set, score).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from . import config


@dataclass
class Pair:
    han_idx: list[int]
    viet_idx: list[int]
    score: float
    method: str          # 'anchor' | 'embed'


# ---- anchors ----------------------------------------------------------------
# Niên hiệu Minh Mệnh + số đếm Hán; bên Việt là 'Minh Mệnh năm thứ ...'
_NIANHAO_HAN = re.compile(r"明命")
_NUM_HAN = "一二三四五六七八九十百千"
_VI_REIGN = re.compile(r"minh\s*mệnh", re.IGNORECASE)


def _han_anchor_key(s: str) -> str | None:
    if _NIANHAO_HAN.search(s):
        nums = "".join(c for c in s if c in _NUM_HAN)
        return "reign:" + nums
    return None


def _viet_anchor_key(s: str) -> str | None:
    if _VI_REIGN.search(s):
        return "reign:*"        # khớp lỏng theo niên hiệu (số đối chiếu sau)
    return None


# ---- embeddings -------------------------------------------------------------
def embed(sents: list[str]):
    from sentence_transformers import SentenceTransformer
    model = embed._model = getattr(embed, "_model", None) or \
        SentenceTransformer(config.LABSE_MODEL)
    return model.encode(sents, convert_to_tensor=True, show_progress_bar=False)


def _sim_matrix(han: list[str], viet: list[str]) -> list[list[float]]:
    from sentence_transformers import util
    Eh, Ev = embed(han), embed(viet)
    return util.cos_sim(Eh, Ev).tolist()


def align(han: list[str], viet: list[str],
          min_sim: float | None = None, max_merge: int | None = None) -> list[Pair]:
    """Quy hoạch động cho phép gộp tối đa max_merge câu mỗi bên (m-n)."""
    min_sim = config.ALIGN_MIN_SIM if min_sim is None else min_sim
    max_merge = config.ALIGN_MAX_MERGE if max_merge is None else max_merge
    if not han or not viet:
        return []
    sim = _sim_matrix(han, viet)

    def block_score(i0, i1, j0, j1):
        # điểm trung bình của khối câu Hán[i0:i1] x Việt[j0:j1]
        vals = [sim[i][j] for i in range(i0, i1) for j in range(j0, j1)]
        return sum(vals) / len(vals) if vals else 0.0

    n, m = len(han), len(viet)
    NEG = -1e9
    # dp[i][j] = điểm tối ưu khi đã dóng xong han[:i], viet[:j]
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] <= NEG:
                continue
            for di in range(0, max_merge + 1):
                for dj in range(0, max_merge + 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if ni > n or nj > m:
                        continue
                    if di == 0 or dj == 0:
                        gain = -0.1  # bỏ trống 1 bên (insertion/deletion)
                    else:
                        gain = block_score(i, ni, j, nj) - 0.05 * (di + dj - 2)
                    if dp[i][j] + gain > dp[ni][nj]:
                        dp[ni][nj] = dp[i][j] + gain
                        back[ni][nj] = (i, j, di, dj, gain)
    # truy vết
    pairs: list[Pair] = []
    i, j = n, m
    while (i, j) != (0, 0):
        bi, bj, di, dj, gain = back[i][j]
        if di > 0 and dj > 0 and gain >= min_sim - 0.1:
            pairs.append(Pair(list(range(bi, i)), list(range(bj, j)),
                              round(gain, 4), "embed"))
        i, j = bi, bj
    pairs.reverse()
    return pairs

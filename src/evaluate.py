"""B6 · Đánh giá dóng hàng câu vs golden bằng Precision / Recall / F1.

golden, pred: tập cặp đã chuẩn hóa thành frozenset cạnh (i_han, j_viet).
Một Pair m-n -> sinh mọi cạnh (i,j) trong khối -> so trên tập cạnh.
"""
from __future__ import annotations


def pairs_to_edges(pairs) -> set[tuple[int, int]]:
    edges = set()
    for p in pairs:
        hi = p["han_idx"] if isinstance(p, dict) else p.han_idx
        vi = p["viet_idx"] if isinstance(p, dict) else p.viet_idx
        for i in hi:
            for j in vi:
                edges.add((i, j))
    return edges


def prf1(pred_edges: set, gold_edges: set) -> dict:
    inter = pred_edges & gold_edges
    P = len(inter) / max(len(pred_edges), 1)
    R = len(inter) / max(len(gold_edges), 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return {"P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
            "pred": len(pred_edges), "gold": len(gold_edges), "hit": len(inter)}

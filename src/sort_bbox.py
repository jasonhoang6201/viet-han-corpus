"""B2-Hán · Sắp bbox theo thứ tự đọc Hán cổ: cột phải→trái, trong cột trên→dưới.

bbox = (x0, y0, x1, y1). 'cx' = tâm x. Gom các bbox có cx gần nhau thành 1 cột,
cột có cx lớn (bên phải) đứng trước; trong cột sắp theo y tăng dần.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def w(self) -> float:
        return abs(self.x1 - self.x0)

    def as_corners(self) -> list[tuple[int, int]]:
        return [(int(self.x0), int(self.y0)), (int(self.x1), int(self.y0)),
                (int(self.x1), int(self.y1)), (int(self.x0), int(self.y1))]


def sort_columns(boxes: list[Box], x_tol: float | None = None) -> list[list[Box]]:
    """Gom -> danh sách cột, cột phải trước. Mỗi cột sắp trên->dưới."""
    if not boxes:
        return []
    if x_tol is None:
        x_tol = 0.6 * (sum(b.w for b in boxes) / len(boxes))  # ~ nửa bề rộng chữ
    bs = sorted(boxes, key=lambda b: -b.cx)  # phải -> trái
    cols: list[list[Box]] = []
    cur: list[Box] = []
    last = None
    for b in bs:
        if last is None or abs(b.cx - last) <= x_tol:
            cur.append(b)
        else:
            cols.append(sorted(cur, key=lambda b: b.y0))
            cur = [b]
        last = b.cx
    if cur:
        cols.append(sorted(cur, key=lambda b: b.y0))
    return cols


def reading_order(boxes: list[Box], x_tol: float | None = None) -> list[Box]:
    """Trả về list phẳng theo đúng thứ tự đọc."""
    return [b for col in sort_columns(boxes, x_tol) for b in col]

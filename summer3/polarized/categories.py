from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summer3.polarized.properties import LazyExpr, PropertyTable

from jax import Array


def cat_prod(a, b):
    out_cats = {}
    for ak, av in a.items():
        for bk, bv in b.items():
            out_cats[f"{ak}__{bk}"] = av & bv
    return out_cats


class CategoryGroup:
    def __init__(self, cats: dict[str, LazyExpr]):
        self.cats = cats

    def is_exclusive(self, pt: PropertyTable) -> bool:
        from summer3.polarized.properties import PropertyTable

        idxset = set()
        for ck, cv in self.cats.items():
            curidx = set(pt.filter(cv).df["index"])
            if len(idxset.intersection(curidx)) > 0:
                return False
            idxset = idxset.union(curidx)
        return True

    def wrap(self, data: Array) -> CategoryData:
        return CategoryData(self, data)

    def __len__(self):
        return len(self.cats)

    def __mul__(self, other):
        if isinstance(other, CategoryGroup):
            return CategoryGroup(cat_prod(self.cats, other.cats))
        else:
            raise TypeError()

    def __repr__(self):
        return f"CategoryGroup:\n{self.cats}"


class CategoryData:
    def __init__(self, cats: CategoryGroup, data: Array):
        assert len(data) == len(cats)
        self.cats = cats
        self.data = data

    def __repr__(self):
        return f"CategoryData:\n{self.cats}\n{self.data}"

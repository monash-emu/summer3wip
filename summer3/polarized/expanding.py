from __future__ import annotations

import polars as pl
import numpy as np
from bidict import bidict
from jax import Array, numpy as jnp

from typing import Sequence
from itertools import product

from summer3.polarized.properties import Property, PropertyTable, LazyExpr
from summer3.polarized.categories import CategoryGroup, CategoryData

# from summer3.polarized.flows import FlowSpec, source, dest


def op_cats_to_idx_series(op_cats: CategoryGroup, fpt):
    op_table = np.ones(len(fpt), dtype=int) * -999

    # Check exclusivity here
    if not op_cats.is_exclusive(fpt):
        raise ValueError(
            "Categories in flow operations must be mutually exclusive", op_cats, fpt
        )
    for i, (_, cat) in enumerate(op_cats.cats.items()):
        catidx = fpt.filter(cat).index.to_numpy()
        op_table[catidx] = i

    return pl.Series(op_table).replace(-999, None)


class ExpandingArray:
    def __init__(self, opidx: pl.Series, data: Array, pt: PropertyTable):
        self.opidx = opidx
        self._data = data
        self.pt = pt

    @property
    def data(self):
        return self._data[self.opidx.to_numpy()]

    def apply_op(self, op: CategoryData):  # , fpt: PropertyTable):
        return apply_op(self, op, self.pt)

    def __repr__(self):
        return f"ExpandingArray\n{self.opidx}\n{self._data}\n{self.data}"

    @classmethod
    def from_scalar(cls, value: float | int, pt: PropertyTable) -> ExpandingArray:
        return scalar_to_expanding(value, pt)


def scalar_to_expanding(value: float | int, pt: PropertyTable) -> ExpandingArray:
    opidx = pl.Series(np.zeros(len(pt), dtype=int))
    return ExpandingArray(opidx, jnp.array((value,)), pt)


def apply_op(lhs: ExpandingArray, so: CategoryData, pt: PropertyTable):
    df_op = pl.DataFrame({"lhs": lhs.opidx, "rhs": op_cats_to_idx_series(so.cats, pt)})

    # Get the rows that are unique pairs
    unique_opdf = df_op.unique()

    null_mask = unique_opdf.with_columns(pl.all().is_null())
    rhs_mask = ~null_mask["rhs"]

    lhs_indices = unique_opdf["lhs"].to_numpy()
    rhs_selector = unique_opdf["rhs"].filter(rhs_mask).to_numpy()

    # These are the 'at' selectors for the expanded LHS (ie where to update)
    active_indices_lhs = np.arange(len(unique_opdf))[rhs_mask]

    new_data = lhs._data[lhs_indices].at[active_indices_lhs].mul(so.data[rhs_selector])

    gwi = unique_opdf.with_columns(
        pl.Series("opidx", np.arange(len(unique_opdf))),
        pl.struct("lhs", "rhs").alias("op_group"),
    ).drop("lhs", "rhs")

    restacked_opidx = df_op.with_columns(
        pl.struct("lhs", "rhs").alias("op_group")
    ).join(gwi, on="op_group")[
        "opidx"
    ]  # .select(pl.col("opidx"))#.transpose(column_names=sd.df.columns)

    return ExpandingArray(restacked_opidx, new_data, pt)


def catdata_to_expanding(cat_data: CategoryData, pt: PropertyTable):
    df = op_cats_to_idx_series(cat_data.cats, pt)
    if df.null_count() > 0:
        raise ValueError("cat_data must cover entire PropertyTable")
    return ExpandingArray(df, cat_data.data, pt)


class OrderedOp:
    def __init__(self, value: float | CategoryData, order: int):
        self.value = value
        self.order = order

    def __repr__(self):
        return f"OrderedOp {self.order}"


class MulOp(OrderedOp):
    def __init__(self, value: float | CategoryData, order: int):
        super().__init__(value, order)


# b = [OrderedOp(*x) for x in a]
# sorted(b, key=lambda x: x.order)

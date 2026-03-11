from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from summer3.polarized.categories import CategoryGroup

import numpy as np
import polars as pl
from bidict import bidict
from typing import Sequence

from summer3.utils import get_unique_keyname


# class PAccessExpr:
#     def __init__(self, pl_expr, pa):
#         self.pl_expr = pl_expr
#         self.pa = pa

#     def __and__(self, other):
#         return PAccessExpr(self.pl_expr & other.pl_expr, self.pa)

#     def __invert__(self):
#         return PAccessExpr(~self.pl_expr, self.pa)

#     def __repr__(self):
#         return f"PAccessExpr[{self.pa}] {self.pl_expr}"


class ActualizedPA:
    def __init__(self, pa, ptable):
        self.pa = pa
        self.prop_key = pa.prop_key
        self._ptable = ptable

        self._uname = uname = ptable._pu[pa.prop_key]
        self._pik_map = ptable.uname_prop_map[uname]._trait_idx_map.inverse

        self.prop_lookup = self._pik_map.inverse

    def __map_args__(self, args):
        try:
            iargs = self._pik_map.inverse[args]
        except:
            iargs = [self._pik_map.inverse[a] for a in args]

        return iargs

    # def _binop(self, other, op):
    #     iarg = self._pik_map.inverse[other]
    #     pl_expr = getattr(pl.col(self._uname), op)(iarg)
    #     return PAccessExpr(pl_expr, self)


class LazyExpr:
    def __and__(self, other):
        return LazyAnd(self, other)

    def __invert__(self):
        return LazyInvert(self)

    def __or__(self, other):
        return LazyOr(self, other)

    def actualize(self, ptable):
        raise NotImplementedError()


class LazyInvert(LazyExpr):
    def __init__(self, expr):
        self.expr = expr

    def actualize(self, ptable):
        return ~self.expr.actualize(ptable)


class LazyAnd(LazyExpr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def actualize(self, ptable):
        return self.lhs.actualize(ptable) & self.rhs.actualize(ptable)

    def __repr__(self):
        return f"({self.lhs} & {self.rhs})"


class LazyOr(LazyExpr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def actualize(self, ptable):
        return self.lhs.actualize(ptable) | self.rhs.actualize(ptable)


class LazyUOp(LazyExpr):
    def __init__(self, pa, op):
        self.pa = pa
        self.op = op

    def actualize(self, ptable):
        apa = ActualizedPA(self.pa, ptable)
        pl_expr = getattr(pl.col(apa._uname), self.op)()
        return pl_expr


class LazyIsIn(LazyExpr):
    def __init__(self, pa, args):
        self.pa = pa
        self.args = args

    def actualize(self, ptable):
        args = self.args
        if isinstance(args, str):
            args = [args]
        apa = ActualizedPA(self.pa, ptable)
        iargs = apa.__map_args__(args)
        return pl.col(apa._uname).is_in(iargs)


class LazyIsBetween(LazyExpr):
    def __init__(self, pa, lower_bound, upper_bound, closed):
        self.pa = pa
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.closed = closed

    def actualize(self, ptable):
        apa = ActualizedPA(self.pa, ptable)
        lower_bound = apa.prop_lookup[self.lower_bound]
        upper_bound = apa.prop_lookup[self.upper_bound]
        return pl.col(apa._uname).is_between(lower_bound, upper_bound, self.closed)


OP_REPR = {
    "eq": "==",
    "ge": ">=",
    "gt": ">",
    "le": "=<",
    "lt": "<",
}


class LazyBinOp(LazyExpr):
    def __init__(self, pa, other, op):
        self.pa = pa
        self.other = other
        self.op = op

    def actualize(self, ptable):
        apa = ActualizedPA(self.pa, ptable)
        iarg = apa._pik_map.inverse[self.other]
        pl_expr = getattr(pl.col(apa._uname), self.op)(iarg)
        return pl_expr

    def __repr__(self):
        return f"{self.pa.prop_key} {OP_REPR[self.op]} {self.other}"


class PropertyAccessor(LazyExpr):
    # prop_key is probably a Stratification for our first example
    def __init__(self, prop_key):  # , ptable: "PropertyTable"):
        self.prop_key = prop_key
        # self._ptable = ptable

        # self._uname = uname = ptable.su[prop_key]
        # self._pik_map = ptable.uname_propidxkey_map[uname]

    def __repr__(self):
        return f"PropertyAccessor[{self.prop_key}]"

    def is_in(self, args):
        return LazyIsIn(self, args)

    def is_between(self, lower_bound: str, upper_bound: str, closed="both"):
        """_summary_

        Args:
            lower_bound: _description_
            upper_bound: _description_
            closed (str, optional): {'both', 'left', 'right', 'none'}

        Returns:
            _type_: _description_
        """
        return LazyIsBetween(self, lower_bound, upper_bound, closed)

    def is_null(self):
        return LazyUOp(self, "is_null")

    def _binop(self, other, op):
        # iarg = self._pik_map.inverse[other]
        # pl_expr = getattr(pl.col(self._uname), op)(iarg)
        return LazyBinOp(self, other, op)

    def eq(self, other):
        return self._binop(other, "eq")

    def __eq__(self, other):
        return self.eq(other)

    def ne(self, other):
        return self._binop(other, "ne")

    def __ne__(self, other):
        return self.ne(other)

    def ge(self, other):
        return self._binop(other, "ge")

    def __ge__(self, other):
        return self.ge(other)

    def gt(self, other):
        return self._binop(other, "gt")

    def __gt__(self, other):
        return self.gt(other)

    def le(self, other):
        return self._binop(other, "le")

    def __le__(self, other):
        return self.le(other)

    def lt(self, other):
        return self._binop(other, "lt")

    def __lt__(self, other):
        return self.lt(other)

    def actualize(self, ptable):
        return (~self.is_null()).actualize(ptable)


class Property(PropertyAccessor):
    def __init__(self, name: str, traits: Sequence[str]):
        super().__init__(self)
        self.name = name
        self.traits = np.array(traits)
        self._trait_idx_map = bidict({self.traits[i]: i for i in range(len(traits))})
        self._uname = None

    def __len__(self):
        return len(self.traits)

    def categories(self) -> CategoryGroup:
        from summer3.polarized.categories import CategoryGroup

        cats = {t: (self == t) for t in self.traits}
        return CategoryGroup(cats)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            traits = self.traits[idx]
        elif isinstance(idx, int):
            traits = self.traits[idx : idx + 1]
        elif isinstance(idx, str):
            traits = np.array([idx])
        else:
            if isinstance(idx[0], int):
                traits = self.traits[idx]
            elif isinstance(idx[0], str):
                # +++
                # This needs better checking - all t must be in our traits etc
                traits = np.array([t for t in idx])
        return PropertyView(self, traits)

    def __repr__(self):
        return f"Property[{self.name}({self._uname})]"

    def __hash__(self):
        return id(self)


class PropertyView:
    def __init__(self, prop: Property, traits: Sequence[str]):
        self.prop = prop
        self.traits = traits

    def __len__(self):
        return len(self.traits)

    def categories(self) -> CategoryGroup:
        from summer3.polarized.categories import CategoryGroup

        cats = {t: (self.prop == t) for t in self.traits}
        return CategoryGroup(cats)  # type: ignore

    def __repr__(self):
        return f"View of {self.prop}: {self.traits}"


class ProxyProperty(Property):
    def __init__(self, existing, prefix):
        self._prefix = prefix
        self._existing = existing
        super().__init__(f"{prefix}({existing.name})", existing.traits)

    def __hash__(self):
        return hash((self._prefix, hash(self._existing)))


UnameIndexKeyMap = dict[str, bidict[int, str]]
UnamePropertyMap = bidict[str, Property]
PropertyAccessorMap = dict[str, PropertyAccessor]


class PropertyTable:
    def __init__(
        self,
        uname_prop_map: UnamePropertyMap,
        prop_df: pl.DataFrame,
        # build_properties: PropertiesBuild = True,
    ):
        """_summary_

        Args:
            uname_prop_map (UnamePropertyMap): uname to Property
            prop_df (pl.DataFrame): Underling polars representation
        """
        self.uname_prop_map = uname_prop_map
        self.df = prop_df

        self._up = self.uname_prop_map
        self._pu = self.uname_prop_map.inverse

        self._str_table = None

    @property
    def index(self):
        return self.df["index"]

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_property(cls, prop: Property):
        uname_prop_map = bidict()
        uname = get_unique_keyname(prop.name, uname_prop_map)
        uname_prop_map[uname] = prop

        df = pl.DataFrame(
            {"index": np.arange(len(prop.traits)), uname: np.arange(len(prop.traits))}
        )

        return cls(uname_prop_map, df)

    def filter(self, expr, rebuild_index=False) -> PropertyTable:
        query_expr = expr.actualize(self)
        filtered_df = self.df.filter(query_expr)
        if rebuild_index:
            filtered_df = filtered_df.with_columns(index=np.arange(len(filtered_df)))
        return PropertyTable(self.uname_prop_map, filtered_df)
        # return self._filter_by_df(filtered_df, rebuild_index)

    def reindex(self) -> PropertyTable:
        return PropertyTable(
            self.uname_prop_map, self.df.with_columns(index=np.arange(len(self.df)))
        )

    def stratify(self, prop: Property, query: LazyExpr | Property):
        # return df.filter(query).with_columns(pl.all().repeat_by(n)).explode(pl.all())
        if prop in self.uname_prop_map.inverse:
            raise KeyError("Property already registered", prop)

        uname = get_unique_keyname(prop.name, self.uname_prop_map)

        if isinstance(query, Property):
            query = ~query.is_null()

        pl_expr = query.actualize(self)

        df_repcounts = self.df.with_columns(
            [
                pl.when(pl_expr)
                .then(pl.lit(len(prop.traits)))
                .otherwise(pl.lit(0))
                .alias("_strat_reps")
            ]
        ).drop("index")
        df_stratified = (
            df_repcounts.with_columns(
                pl.int_ranges(0, pl.col("_strat_reps")).alias(uname)
            )
            .drop("_strat_reps")
            .explode(uname)
        )

        new_df = df_stratified.with_columns(index=np.arange(len(df_stratified)))
        new_up_map = self.uname_prop_map | {uname: prop}

        return PropertyTable(new_up_map, new_df)

    @property
    def named_df(self):
        return self._named_table()

    def _named_table(self):
        if self._str_table is None:
            self._str_table = self.df.with_columns(
                [
                    pl.col(col).replace_strict(
                        self._up[col]._trait_idx_map.inverse, return_dtype=pl.String
                    )
                    for col in self._up
                ]
            )
        return self._str_table

    def __repr__(self):
        return f"PropertyTable\n{self._named_table()}"

    def _repr_html_(self):
        return f"PropertyTable<br>{self._named_table()._repr_html_()}"


def concat_pt(prop_tables: Sequence[PropertyTable], reindex=True) -> PropertyTable:
    ref_pt = prop_tables[0]
    for pt in prop_tables[1:]:
        if (rupm := ref_pt.uname_prop_map) != (cupm := pt.uname_prop_map):
            raise ValueError(
                "PropertyTables must have identical uname_prop_map", rupm, cupm
            )
    df = pl.concat([pt.df for pt in prop_tables])
    if reindex:
        df = df.with_columns(index=np.arange(len(df)))
    return PropertyTable(ref_pt.uname_prop_map, df)


class AccessorStrat(PropertyAccessor):
    def __init__(self, name, strata):
        self.prop_key = self
        self.name = name
        self.strata = strata

    def _set_ptable(self, ptable):
        self.ptable = ptable

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"AccessorStrat[{self.name}]"


def build_astrat(name, strata):
    uname_strat_map = bidict()
    uname_propidxkey_map: dict[str, bidict] = {}
    _prop_tab = {}

    strat = AccessorStrat(name, strata)

    uname = get_unique_keyname(name, uname_strat_map)
    uname_strat_map[uname] = strat
    _prop_tab[uname] = v = np.arange(len(strat.strata), dtype=int)
    uname_propidxkey_map[uname] = bidict({i: k for (i, k) in enumerate(strat.strata)})

    prop_table = pl.DataFrame(
        _prop_tab | {"index": np.arange(len(strat.strata), dtype=int)}
    )

    ptable = PropertyTable(uname_strat_map, uname_propidxkey_map, prop_table, False)
    ptable.properties = {list(uname_strat_map.inverse.values())[0]: strat}

    strat._set_ptable(ptable)

    return strat


def strat_to_prop_table(strat):
    uname_strat_map = bidict()
    uname_propidxkey_map: dict[str, bidict] = {}
    _prop_tab = {}

    name = strat.name
    uname = get_unique_keyname(name, uname_strat_map)
    uname_strat_map[uname] = strat
    _prop_tab[uname] = v = np.arange(len(strat.strata), dtype=int)
    uname_propidxkey_map[uname] = bidict({i: k for (i, k) in enumerate(strat.strata)})

    prop_table = pl.DataFrame(
        _prop_tab | {"index": np.arange(len(strat.strata), dtype=int)}
    )

    return PropertyTable(uname_strat_map, uname_propidxkey_map, prop_table)


def build_property_tables(cm, compartments):
    # Stratifications do not require unique names
    # but we need unique strings for non-object supporting
    # table keys (polars etc)

    # Map uname to accessor key
    uname_strat_map = bidict()

    # Questionable - this could live in the accessors themselves
    # (ie it is just the list of strata mapped to integers)
    uname_propidxkey_map: dict[str, bidict] = {}
    _prop_tab = {}

    # stratify_with
    # adds a uname
    # updates the map
    # adds a column to the table

    for strat in cm.stratifications:
        name = strat.name
        uname = get_unique_keyname(name, uname_strat_map)
        uname_strat_map[uname] = strat
        _prop_tab[uname] = v = np.empty(len(compartments), dtype=int)
        v.fill(-1)
        uname_propidxkey_map[uname] = bidict(
            {i: k for (i, k) in enumerate(strat.strata)}
        )

    for comp_i, c in enumerate(compartments):
        for strat, stratum in c.strata:
            uname = uname_strat_map.inverse[strat]
            ik_map = uname_propidxkey_map[uname]
            prop_i = ik_map.inverse[stratum]
            _prop_tab[uname][comp_i] = prop_i

    prop_table = pl.DataFrame(
        _prop_tab | {"index": np.arange(len(compartments))}
    ).with_columns(pl.all().replace(-1, None))

    return PropertyTable(uname_strat_map, uname_propidxkey_map, prop_table)

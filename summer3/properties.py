import numpy as np
import polars as pl
from bidict import bidict

from summer3.utils import get_unique_keyname


class PAccessExpr:
    def __init__(self, pl_expr, pa):
        self.pl_expr = pl_expr
        self.pa = pa

    def __and__(self, other):
        return PAccessExpr(self.pl_expr & other.pl_expr, self.pa)

    def __invert__(self):
        return PAccessExpr(~self.pl_expr, self.pa)

    def __repr__(self):
        return f"PAccessExpr[{self.pa}] {self.pl_expr}"


class PropertyAccessor:
    # prop_key is probably a Stratification for our first example
    def __init__(self, prop_key, ptable: "PropertyTable"):
        self.prop_key = prop_key
        self._ptable = ptable

        self._uname = uname = ptable.su[prop_key]
        self._pik_map = ptable.uname_propidxkey_map[uname]

    def __repr__(self):
        return f"PropertyAccessor[{self.prop_key}]"

    def __map_args__(self, args):
        try:
            iargs = self._pik_map.inverse[args]
        except:
            iargs = [self._pik_map.inverse[a] for a in args]

        return iargs

    def is_in(self, args):
        if isinstance(args, str):
            args = [args]
        iargs = self.__map_args__(args)
        pl_expr = pl.col(self._uname).is_in(iargs)
        return PAccessExpr(pl_expr, self)

    def _binop(self, other, op):
        iarg = self._pik_map.inverse[other]
        pl_expr = getattr(pl.col(self._uname), op)(iarg)
        return PAccessExpr(pl_expr, self)

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


class PropertyTable:
    def __init__(self, uname_strat_map, uname_propidxkey_map, prop_table):
        self.uname_strat_map = uname_strat_map
        self.uname_propidxkey_map = uname_propidxkey_map
        self.table = prop_table

        self.us = self.uname_strat_map
        self.su = self.uname_strat_map.inverse

        self.properties = self.__build_accessors__()

    def __build_accessors__(self):
        return {k: PropertyAccessor(k, self) for k in self.uname_strat_map.inverse}


def build_property_tables(cm, compartments):
    # Stratifications do not require unique names
    # but we need unique strings for non-object supporting
    # table keys (polars etc)
    uname_strat_map = bidict()
    uname_propidxkey_map: dict[str, bidict] = {}
    _prop_tab = {}
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

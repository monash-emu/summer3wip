import polars as pl
import numpy as np
from bidict import bidict

from typing import Sequence

from summer3.polarized.properties import (
    Property,
    ProxyProperty,
    PropertyTable,
    LazyExpr,
)
from summer3.polarized.categories import CategoryGroup, CategoryData
from summer3.polarized.properties import PropertyTable
from summer3.polarized.expanding import OrderedOp, MulOp

from summer3.computegraph.types import GraphObject


def source(prop: Property) -> ProxyProperty:
    return ProxyProperty(prop, "source")


def dest(prop: Property) -> ProxyProperty:
    return ProxyProperty(prop, "dest")


class TransitionFlowResolver:
    def __init__(
        self,
        sourceq: LazyExpr | CategoryGroup,
        destq: LazyExpr | CategoryGroup,
        pt: PropertyTable,
    ):
        self.sourceq = sourceq
        self.destq = destq
        self.pt = pt
        self._initialized = False

    def _init_internals(self):
        if not self._initialized:
            self.source_table = self.pt.filter(self.sourceq, rebuild_index=False)
            self.dest_table = self.pt.filter(self.destq, rebuild_index=False)

            self.source_df = self.source_table.df
            self.dest_df = self.dest_table.df

            self.source_props = source_props = valid_columns(self.source_df)
            self.dest_props = dest_props = valid_columns(self.dest_df)

            self.common_props = list(set(source_props).intersection(set(dest_props)))

            self.erased_props = list(set(source_props).difference(set(dest_props)))
            self.created_props = list(set(dest_props).difference(set(source_props)))
            self._initialized = True

    def get_policy_tables(self):
        props_matched = []  # A->A
        props_moved = []  # Nothing in common (moved)
        props_to_map = {}

        self._init_internals()

        for column in self.common_props:
            source_props = self.source_df[column].unique()
            dest_props = self.dest_df[column].unique()

            ns = len(source_props)
            nd = len(dest_props)

            if ns == nd:
                if ns == 1:
                    props_moved.append(column)
                elif set(source_props) != set(dest_props):
                    raise Exception("Multimap not supported yet", column)
                else:
                    props_matched.append(column)
            elif ns != nd:
                if ns == 1:
                    # One to many
                    props_to_map[column] = RMap(
                        column, {v: source_props.item() for v in dest_props}
                    )
                elif nd == 1:
                    # Many to one
                    props_to_map[column] = LMap(
                        column, {v: dest_props.item() for v in source_props}
                    )
                else:
                    print(column, ns, nd)
                    raise Exception("Multimap not supported yet", column)
            # else:
            #    if (source_props == dest_props).all():
            #        props_matched.append(column)
            #    elif (source_props != dest_props).all():
            #        props_moved.append(column)

        return {"matched": props_matched, "moved": props_moved, "mapped": props_to_map}

    def get_flow_pt(self) -> PropertyTable:
        fs = self
        if isinstance(fs.sourceq, CategoryGroup):
            if isinstance(fs.destq, CategoryGroup):
                if not len(fs.sourceq.cats) == len(fs.destq.cats):
                    raise ValueError("CategoryGroup length mismatch")
                accum_df = []
                for (sk, sv), (dk, dv) in zip(
                    fs.sourceq.cats.items(), fs.destq.cats.items()
                ):
                    pt = TransitionFlowResolver(sv, dv, fs.pt).get_flow_pt()
                    accum_df.append(pt.df)
                full_df = pl.concat(accum_df)
                full_df = full_df.with_columns(index=np.arange(len(full_df)))
                return PropertyTable(pt.uname_prop_map, full_df)
            else:
                raise TypeError("sourceq and destq must be of same type")

        policy = fs.get_policy_tables()

        source_df = fs.source_df
        dest_df = fs.dest_df

        orig_cols = source_df.columns

        prop_names = [k for k in source_df.columns if k != "index"]

        mapped_cols = []

        for col, mapping in policy["mapped"].items():
            map_col, source_df, dest_df = mapping.map(source_df, dest_df)
            mapped_cols.append(map_col)

        jgroups = policy["matched"] + mapped_cols

        sj = source_df.with_columns(pl.struct(jgroups).alias("jgroup")).drop(
            mapped_cols
        )
        dj = dest_df.with_columns(pl.struct(jgroups).alias("jgroup")).drop(mapped_cols)

        joined = sj.join(dj, on="jgroup", suffix="_dest").drop("jgroup")
        joined = joined.rename({k: f"{k}_source" for k in orig_cols})

        source_prop_map = {
            f"{k}_source": source(fs.pt.uname_prop_map[k]) for k in prop_names
        }

        dest_prop_map = {f"{k}_dest": dest(fs.pt.uname_prop_map[k]) for k in prop_names}

        full_uname_prop_map = bidict(source_prop_map | dest_prop_map)

        joined = joined.with_columns(index=np.arange(len(joined)))

        return PropertyTable(full_uname_prop_map, joined)  # type: ignore


class LMap:
    def __init__(self, col, mapping):
        self.col = col
        self.mapping = mapping

    def map(self, source, dest):
        col = self.col
        map_col = f"{col}__mapped"
        source_mapped = source.with_columns(
            pl.col(col).replace(self.mapping).alias(map_col)
        )
        dest_mapped = dest.with_columns(pl.col(col).alias(map_col))

        return map_col, source_mapped, dest_mapped


class RMap:
    def __init__(self, col, mapping):
        self.col = col
        self.mapping = mapping

    def map(self, source, dest):
        col = self.col
        map_col = f"{col}__mapped"
        dest_mapped = dest.with_columns(
            pl.col(col).replace(self.mapping).alias(map_col)
        )
        source_mapped = source.with_columns(pl.col(col).alias(map_col))

        return map_col, source_mapped, dest_mapped


def valid_columns(df):
    validity_s = df.select(~pl.all().is_null().all())
    valid_columns = [c.name for c in validity_s if c[0] and (c.name != "index")]
    return valid_columns


from summer3.polarized.properties import PropertyTable


class FlowMap:
    def build_flow_pt(self, comp_pt: PropertyTable) -> PropertyTable:
        raise NotImplementedError()


class ExitMap(FlowMap):
    def __init__(self, source_query: LazyExpr):
        self.source_query = source_query

    def build_flow_pt(self, comp_pt: PropertyTable) -> PropertyTable:
        """Build a PropertyTable representing this flow
        Args:
            comp_pt: Compartments PropertyTable

        Returns:
            The flow PropertyTable (with source properties)
        """
        source_table = comp_pt.filter(self.source_query, rebuild_index=False)
        source_df = source_table.df

        prop_names = [k for k in source_df.columns if k != "index"]
        source_prop_map = {
            f"{k}_source": source(comp_pt.uname_prop_map[k]) for k in prop_names
        }

        new_df = source_df.rename(
            {k: f"{k}_source" for k in source_df.columns}
        ).with_columns(index=np.arange(len(source_df)))

        new_uname_prop_map = bidict(source_prop_map)
        return PropertyTable(new_uname_prop_map, new_df)  # type: ignore


class EntryMap(FlowMap):
    def __init__(self, dest_query: LazyExpr):
        self.dest_query = dest_query

    def build_flow_pt(self, comp_pt: PropertyTable) -> PropertyTable:
        """Build a PropertyTable representing this flow
        Args:
            comp_pt: Compartments PropertyTable

        Returns:
            The flow PropertyTable (with source properties)
        """
        dest_table = comp_pt.filter(self.dest_query, rebuild_index=False)
        dest_df = dest_table.df

        prop_names = [k for k in dest_df.columns if k != "index"]
        dest_prop_map = {
            f"{k}_dest": dest(comp_pt.uname_prop_map[k]) for k in prop_names
        }

        new_df = dest_df.rename({k: f"{k}_dest" for k in dest_df.columns}).with_columns(
            index=np.arange(len(dest_df))
        )

        new_uname_prop_map = bidict(dest_prop_map)
        return PropertyTable(new_uname_prop_map, new_df)  # type: ignore


class TransitionMap(FlowMap):
    def __init__(
        self,
        source_query: LazyExpr | CategoryGroup,
        dest_query: LazyExpr | CategoryGroup,
    ):
        self.source_query = source_query
        self.dest_query = dest_query

    def build_flow_pt(self, comp_pt: PropertyTable) -> PropertyTable:
        fspec = TransitionFlowResolver(self.source_query, self.dest_query, comp_pt)
        return fspec.get_flow_pt()


# +++
# We need to specify absolute/proportional somewhere - presumably on Flow class constructor
# Which always applies to source compartments, so entry flows are always absolute
# Notes:
# Need to provide actualize_flow method (easy enough - get the indices from the model PT and use expanding ops
# to perform adjustments and final multiply
#


# FlowSpec
# fmap
# base_param
# operations

# TransitionFlow
# fmap : TransitionMap
# operations: [BaseOp(base_param, order=FIRST), MulOp]


class FlowSpec:
    def __init__(
        self,
        name: str,
        fmap: FlowMap,
        initial_ops: list[OrderedOp] | float | CategoryData | GraphObject,
        # base_param: float | CategoryData | cg.types.GraphObject,
    ):
        self.name = name
        self.fmap = fmap
        if not isinstance(initial_ops, list):
            initial_ops = [MulOp(initial_ops, 0)]
        self.operations: list[OrderedOp] = initial_ops
        # self.meta = {}  +++ We probably will need this for advanced flow queries later - but let's not
        # get ahead of ourselves...

    def add_op(self, op: OrderedOp):
        self.operations = sorted(self.operations + [op], key=lambda x: x.order)

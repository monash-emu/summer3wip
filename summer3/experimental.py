from jax import numpy as jnp
import numpy as np
from .proto import CompartmentContainer, get_cat_indices, StratSpec, CategoryGroup
import pandas as pd
from numbers import Integral
from typing import Optional
from .utils import get_category_names


class LA:
    def __init__(self, data, axes):
        self.data = data
        self.axes = axes

        self._ax_to_idx = {k: i for i, k in enumerate(axes)}
        self._idx_to_ax = {v: k for k, v in self._ax_to_idx.items()}

    def transpose(self, axes):
        transposed = self.data.transpose([self._ax_to_idx[a] for a in axes])
        return LA(transposed, axes)

    def expand(self, ax, index=-1):
        expanded = jnp.expand_dims(self.data, index)
        if index == -1:
            new_axes = self.axes + [ax]
        else:
            new_axes = []
            for a, i in enumerate(self.axes):
                if i == index:
                    new_axes.append(ax)
                new_axes.append(a)
        return LA(expanded, new_axes)

    def reconcile(self, other):
        s_set = set(self.axes)
        o_set = set(other.axes)
        s_extras = s_set.difference(o_set)
        o_extras = o_set.difference(s_set)
        other_expanded = other
        self_expanded = self
        for eax in list(s_extras):
            other_expanded = other_expanded.expand(eax)
        for eax in list(o_extras):
            self_expanded = self_expanded.expand(eax)
        return self_expanded, other_expanded.transpose(self_expanded.axes)

    def to_axis(self, ax):
        if ax not in self.axes:
            raise KeyError("Axis not found", ax)
        return [a for a in self.axes if a != ax]

    def __mul__(self, other):
        return self._lop(other, jnp.multiply)

    def __rmul__(self, other):
        return self._rop(other, jnp.multiply)

    def __truediv__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data / orec.data, srec.axes)

    def __rtruediv__(self, other):
        return self._rop(other, jnp.true_divide)

    def _lop(self, other, op):
        if isinstance(other, float):
            return LA(op(self.data, other), self.axes)
        srec, orec = self.reconcile(other)
        return LA(op(srec.data, orec.data), srec.axes)

    def _rop(self, other, op):
        if isinstance(other, float):
            return LA(op(other, self.data), self.axes)
        else:
            raise TypeError("Unsupported type", other)

    def __add__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data + orec.data, srec.axes)

    def __sub__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data - orec.data, srec.axes)

    def sum(self, axis=None):
        return self._liftreduction("sum", axis)

    def _reduce_axes(self, axis=None):
        if axis is None:
            lifted_ax = None
            reduced_axes = self.axes
        elif isinstance(axis, str):
            lifted_ax = self._ax_to_idx[axis]
            reduced_axes = [a for a in self.axes if a != axis]
        else:
            lifted_ax = [self._ax_to_idx[a] for a in axis]
            reduced_axes = [a for a in self.axes if a not in axis]
        return lifted_ax, reduced_axes

    def _liftreduction(self, op, axis=None):
        lifted_ax, reduced_axes = self._reduce_axes(axis)
        return LA(getattr(self.data, op)(axis=lifted_ax), reduced_axes)

    def __repr__(self):
        data_repr = repr(self.data)
        info_repr = f"LA {self.axes} {self.data.shape}\n"
        return info_repr + data_repr


class ManagedIndex:
    def __init__(self, dim, index):
        self.dim = dim
        self.index = index

    def __repr__(self):
        return f"ManagedIndex: maps {self.dim}\n" + repr(self.index)

    def query(self, q):
        if isinstance(self.index, CompartmentContainer):
            qres = self.index.query(q)
            new_subidx, idx_arr = qres, qres.parent_indices
        elif isinstance(self.index, pd.Index):
            pdlookup = pd.Series(index=self.index, data=np.arange(len(self.index)))
            qbackref = pdlookup[q]
            if isinstance(qbackref, Integral):
                qbackref = pdlookup[[q]]
            new_subidx, idx_arr = qbackref.index, np.array(qbackref)
        else:
            raise TypeError(self.index)
        return ManagedIndex(self.dim, new_subidx), squash_to_slice(idx_arr)


class ManagedCategoryGroupIndex(ManagedIndex):
    def __init__(self, dim: str, index: CategoryGroup):
        super().__init__(dim, index)

    def query(self, q):
        qres = self.index.query(q)
        return ManagedCategoryGroupIndex(self.dim, qres, squash_to_slice(qres.indices))

    def __repr__(self):
        return f"ManagedCategoryGroupIndex: maps [{self.dim}]\n" + repr(self.index)


class ManagedArray:
    def __init__(
        self,
        data,
        dims,
        indices: Optional[dict[str, ManagedIndex]] = None,
        labellers=None,
    ):
        self.data = data
        self._la = LA(data, dims)
        self.dims = dims
        self._dim_idx = {dim: i for i, dim in enumerate(dims)}
        self.indices = indices or {}
        self.labellers = labellers or {}

    def add_index(self, name, dim, index):
        self.indices[name] = ManagedIndex(dim, index)

    @property
    def shape(self):
        return self.data.shape

    def query(self, **kwargs):
        qindices = []
        for idx_name, q in kwargs.items():
            mindex = self.indices[idx_name]
            di = self._dim_idx[mindex.dim]
            new_subidx, qidx = mindex.query(
                q
            )  # self._handle_index_query(mindex.index, q)
            qindices.append((di, (idx_name, new_subidx, qidx)))
        # qindices = sorted(qindices, key=lambda x: x[0])
        qindices = {dimi: q for dimi, q in qindices}
        slicer = []
        new_indices = {}
        for i in range(len(self.dims)):
            if i in qindices:
                idx_name, new_subidx, qidx = qindices[i]
                new_indices[idx_name] = new_subidx
                slicer.append(qidx)
            else:
                slicer.append(...)
        for k, v in self.indices.items():
            if k not in new_indices:
                new_indices[k] = v
        try:
            out_data = self.data[*slicer]
            return ManagedArray(out_data, self.dims, new_indices)
        except:
            raise Exception("Unsupported slice styles; try chaining queries")

    def __repr__(self):
        return "ManagedArray\n" + repr(self.dims) + repr(self.indices) + repr(self.data)

    def sumcats(self, index_name, categories: CategoryGroup):
        idx_name, catgroups = index_name, categories
        indexer = self.indices[idx_name]
        maps_dim, cat_cmap = indexer.dim, indexer.index
        cat_indices = get_cat_indices(catgroups, cat_cmap)
        # cat_names = get_category_names(catgroups)
        dim_idx = self._dim_idx[maps_dim]
        if dim_idx == (len(self.dims) - 1):
            slicers = [slice() for i in range(len(self.dims) - 1)]

            if len(set([len(c) for c in cat_indices])) == 1:
                slicers.append(np.array(cat_indices))
                new_data = self.data[*slicers].sum(axis=-1)
            else:
                new_data = jnp.array(
                    [self.data[*([slicers] + [c])].sum(axis=-1) for c in cat_indices]
                )
        else:
            raise Exception("Only timecubes supported currently")
        out_indices = {
            name: midx for name, midx in self.indices.items() if midx.dim != maps_dim
        }
        out_indices["category"] = ManagedCategoryGroupIndex("category", catgroups)
        return ManagedArray(new_data, ["time", "category"], indices=out_indices)

    def sum(self, cats=None, dims=None):
        if cats is not None:
            idx_name, catgroups = cats
            indexer = self.indices[idx_name]
            maps_dim, cat_cmap = indexer.dim, indexer.index
            cat_indices = get_cat_indices(catgroups, cat_cmap)
            # cat_names = get_category_names(catgroups)
            dim_idx = self._dim_idx[maps_dim]
            if dim_idx == 1 and len(self.dims) == 2:
                if len(set([len(c) for c in cat_indices])) == 1:
                    new_data = self.data[:, np.array(cat_indices)].sum(axis=-1)
                else:
                    new_data = jnp.array(
                        [self.data[:, c].sum(axis=-1) for c in cat_indices]
                    )
            else:
                raise Exception("Only timecubes supported currently")
            out_indices = {
                name: midx
                for name, midx in self.indices.items()
                if midx.dim != maps_dim
            }
            out_indices["category"] = ManagedCategoryGroupIndex(
                "category", CategoryGroup(catgroups)
            )
            return ManagedArray(new_data, ["time", "category"], indices=out_indices)

        if dims is not None:
            lma = LA(self.data, self.dims)
            lsummed = lma.sum(dims)
            out_indices = {
                name: midx
                for name, midx in self.indices.items()
                if midx.dim in lsummed.axes
            }
            return ManagedArray(lsummed.data, lsummed.axes, out_indices)

    def to_pandas_df(self):
        if len(self.dims) != 2:
            raise Exception("Only 2d ManagedArrays supported for Pandas export")
        data_dim = self.dims[1]
        if data_dim in self.labellers:
            labeller = self.labellers[data_dim]
            columns = labeller(self)
        elif data_dim in self.indices:
            col_idx = self.indices[data_dim].index
            if isinstance(col_idx, CompartmentContainer):
                columns = col_idx.get_labels()
            else:
                columns = col_idx
        else:
            columns = None
        return pd.DataFrame(
            index=self.indices["time"].index, data=self.data, columns=columns
        )


def _squash_to_slice(idx_arr):
    # Flat, contiguous
    if (idx_arr[-1] - idx_arr[0]) == (len(idx_arr) - 1):
        if (idx_arr == np.arange(idx_arr[0], idx_arr[-1] + 1)).all():
            return slice(idx_arr[0], idx_arr[-1] + 1)
    # Stepped slice
    diffs = np.diff(idx_arr)
    if len(set(diffs)) == 1:
        step = diffs[0]
        return slice(idx_arr[0], idx_arr[-1] + step, step)

    return idx_arr

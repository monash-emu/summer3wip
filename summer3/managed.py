"""Managed arrays and indices for handling multi-dimensional labeled data.

This module provides classes for working with multi-dimensional arrays that have
labeled dimensions and indices, supporting operations like querying, arithmetic,
and aggregation while preserving dimension semantics.
"""

from __future__ import annotations

from typing import Optional, Callable
from types import ModuleType
from numbers import Integral, Number
from jax import numpy as jnp, Array
import numpy as np
from . import proto
import pandas as pd
from .utils import squash_to_slice, Indexer, get_rolling_reduction
from summer3.polarized.properties import PropertyTable


class ManagedIndex:
    """An index that maps to a specific dimension in a ManagedArray.

    ManagedIndex associates an index (either pandas Index or CompartmentContainer)
    with a dimension name, enabling labeled access to array data.

    Attributes:
        dim (str): The dimension name this index maps to.
        index (pd.Index | proto.CompartmentContainer): The underlying index data.
    """

    def __init__(self, dim, index):
        """Initialize a ManagedIndex.

        Args:
            dim (str): The dimension name this index maps to.
            index (pd.Index | proto.CompartmentContainer): The index data structure.
        """
        self.dim = dim
        self.index = index

    def __repr__(self):
        """Return string representation of the ManagedIndex.

        Returns:
            str: Formatted string showing dimension and index details.
        """
        return f"ManagedIndex: maps {self.dim}\n" + repr(self.index)

    def query(self, q) -> tuple[ManagedIndex, Indexer]:
        """Query the index to create a subset.

        Args:
            q: Query specification. Format depends on index type:
                - For pd.Index: list of values to select
                - For CompartmentContainer: StratSpec query

        Returns:
            tuple[ManagedIndex, Indexer]: A new ManagedIndex with the subset
                and an indexer (slice or array) for the original data.

        Raises:
            TypeError: If the index type is not supported.
        """
        if isinstance(self.index, proto.CompartmentContainer):
            qres = self.index.query(q)
            new_subidx, idx_arr = qres, qres.parent_indices
        elif isinstance(self.index, PropertyTable):
            filtered_pt = self.index.filter(q)
            idx_arr = filtered_pt.df["index"].to_numpy()
            new_subidx = filtered_pt.reindex()
        elif isinstance(self.index, pd.Index):
            pdlookup = pd.Series(index=self.index, data=np.arange(len(self.index)))
            qbackref = pdlookup.loc[q]
            if isinstance(qbackref, Integral):
                qbackref = pdlookup.loc[[q]]
            new_subidx, idx_arr = qbackref.index, np.array(qbackref)
        else:
            raise TypeError(self.index)
        return ManagedIndex(self.dim, new_subidx), squash_to_slice(idx_arr)


class ManagedArray:
    """A multi-dimensional array with labeled dimensions and indices.

    ManagedArray provides a higher-level interface for working with multi-dimensional
    arrays where each dimension has a name and can be associated with indices for
    labeled access. Supports arithmetic operations, querying, aggregation, and
    dimension manipulation while preserving semantic meaning.

    Attributes:
        data (jnp.ndarray): The underlying JAX array data.
        dims (list[str]): Names of the dimensions in order.
        indices (dict[str, ManagedIndex]): Named indices for labeled access.
        labellers (dict): Functions for generating labels for dimensions.
        parent_indices: Indices into a parent array if this is a view.
        shape (tuple[int, ...]): Shape of the underlying data array.
    """

    def __init__(
        self,
        data,
        dims,
        indices: Optional[dict[str, ManagedIndex]] = None,
        labellers=None,
        parent_indices=None,
    ):
        """Initialize a ManagedArray.

        Args:
            data (jnp.ndarray): The array data.
            dims (list[str]): Dimension names, must match data.ndim.
            indices (dict[str, ManagedIndex], optional): Named indices for dimensions.
            labellers (dict, optional): Functions for generating dimension labels.
            parent_indices (optional): Parent array indices if this is a view.

        Raises:
            ValueError: If len(dims) doesn't match data.ndim.
        """
        if len(dims) != len(data.shape):
            raise ValueError(
                f"Shape mismatch between dims {len(dims)} and data {len(data.shape)}"
            )

        self.data = data
        self.dims = dims

        self._dim_idx = {dim: i for i, dim in enumerate(dims)}
        self.indices = indices or {}
        self.labellers = labellers or {}
        self.parent_indices = parent_indices

    def add_index(self, name, dim, index):
        """Add a named index for a dimension.

        Args:
            name (str): Name for the index.
            dim (str): Dimension name the index maps to.
            index (pd.Index | proto.CompartmentContainer): The index data.
        """
        self.indices[name] = ManagedIndex(dim, index)

    def copy_with(self, **kwargs):
        """Create a copy with optionally modified attributes.

        Args:
            **kwargs: Attributes to override in the copy. If not provided,
                     the current values are copied.

        Returns:
            ManagedArray: A new ManagedArray instance.
        """
        out_kwargs = kwargs.copy()
        if "data" not in out_kwargs:
            out_kwargs["data"] = self.data.copy()
        if "dims" not in out_kwargs:
            out_kwargs["dims"] = self.dims.copy()
        if "indices" not in out_kwargs:
            out_kwargs["indices"] = self.indices.copy()
        if "labellers" not in out_kwargs:
            out_kwargs["labellers"] = self.labellers.copy()
        if "parent_indices" not in out_kwargs:
            out_kwargs["parent_indices"] = self.parent_indices
        return ManagedArray(**out_kwargs)

    def simplify(self):
        """Remove dimensions of size 1 from the array.

        Returns:
            ManagedArray: A new array with singleton dimensions removed.
        """
        out_dims = []
        out_shape = []
        for dim, dlen in zip(self.dims, self.shape):
            if dlen != 1:
                out_dims.append(dim)
                out_shape.append(dlen)
        out_data = self.data.reshape(tuple(out_shape))
        out_indices = {k: v for k, v in self.indices.items() if v.dim in out_dims}
        return self.copy_with(data=out_data, dims=out_dims, indices=out_indices)

    def transpose(self, dims):
        """Transpose the array to a new dimension order.

        Args:
            dims (list[str]): New dimension order. Must contain exactly
                             the same dimensions as the current array.

        Returns:
            ManagedArray: A new array with dimensions reordered.

        Raises:
            Exception: If dims doesn't match current dimensions exactly.
        """
        if set(dims) != set(self.dims):
            raise Exception("Dimensions must match exactly")
        transposed_data = self.data.transpose([self._dim_idx[d] for d in dims])
        return ManagedArray(transposed_data, dims, self.indices, self.labellers)

    def expand(self, dim, index=-1):
        """Add a new dimension of size 1 to the array.

        Args:
            dim (str): Name for the new dimension.
            index (int, optional): Position to insert the dimension.
                                 -1 means append to the end.

        Returns:
            ManagedArray: A new array with the added dimension.
        """
        expanded = jnp.expand_dims(self.data, index)
        if index == -1:
            new_axes = self.dims + [dim]
        else:
            new_axes = []
            for i, a in enumerate(self.dims):
                if i == index:
                    new_axes.append(dim)
                new_axes.append(a)
        return ManagedArray(expanded, new_axes, self.indices, self.labellers)

    def reconcile(self, other):
        """Reconcile dimensions between this array and another for broadcasting.

        Expands both arrays to have the same set of dimensions by adding
        singleton dimensions where needed.

        Args:
            other (ManagedArray): The other array to reconcile with.

        Returns:
            tuple[ManagedArray, ManagedArray]: Both arrays expanded to have
                                              the same dimensions.
        """
        s_set = set(self.dims)
        o_set = set(other.dims)

        # +++ Should attempt some kind of matching here...
        # for cdim in s_set.intersection(o_set):
        #    if self.indices[cdim].index != other.indices[cdim].index

        s_extras = s_set.difference(o_set)
        o_extras = o_set.difference(s_set)
        other_expanded = other
        self_expanded = self
        for eax in list(s_extras):
            other_expanded = other_expanded.expand(eax)
        for eax in list(o_extras):
            self_expanded = self_expanded.expand(eax)
        return self_expanded, other_expanded.transpose(self_expanded.dims)

    def _lop(self, other, op):
        """Apply a left binary operation.

        Args:
            other (Number | ManagedArray): Right operand.
            op (callable): Binary operation function.

        Returns:
            ManagedArray: Result of the operation.

        Raises:
            TypeError: If other is not a supported type.
        """
        if isinstance(other, Number):
            return self.copy_with(
                data=op(self.data, other)
            )  # , self.dims, self.indices, self.labellers)
        elif isinstance(other, ManagedArray):
            srec, orec = self.reconcile(other)
            return srec.copy_with(data=op(srec.data, orec.data))
            # return ManagedArray(op(srec.data, orec.data), srec.dims, srec.indices, srec.labellers)
        else:
            raise TypeError("Unsupported type", other)

    def _rop(self, other, op):
        """Apply a right binary operation.

        Args:
            other (Number): Left operand.
            op (callable): Binary operation function.

        Returns:
            ManagedArray: Result of the operation.

        Raises:
            TypeError: If other is not a Number.
        """
        if isinstance(other, Number):
            return self.copy_with(data=op(other, self.data))  # , self.axes)
        else:
            raise TypeError("Unsupported type", other)

    def __mul__(self, other):
        """Element-wise multiplication.

        Args:
            other (Number | ManagedArray): Right operand.

        Returns:
            ManagedArray: Result of multiplication.
        """
        return self._lop(other, jnp.multiply)

    def __rmul__(self, other):
        """Right multiplication (other * self).

        Args:
            other (Number): Left operand.

        Returns:
            ManagedArray: Result of multiplication.
        """
        return self._rop(other, jnp.multiply)

    def __truediv__(self, other):
        return self._lop(other, jnp.true_divide)

    def __floordiv__(self, other):
        return self._lop(other, jnp.floor_divide)

    def __add__(self, other):
        """Element-wise addition.

        Args:
            other (Number | ManagedArray): Right operand.

        Returns:
            ManagedArray: Result of addition.
        """
        return self._lop(other, jnp.add)

    def __radd__(self, other):
        """Right addition (other + self).

        Args:
            other (Number): Left operand.

        Returns:
            ManagedArray: Result of addition.
        """
        return self._rop(other, jnp.add)

        srec, orec = self.reconcile(other)
        return srec.copy_with(data=srec.data + orec.data)

    def __sub__(self, other):
        """Element-wise subtraction.

        Args:
            other (Number | ManagedArray): Right operand.

        Returns:
            ManagedArray: Result of subtraction.
        """
        return self._lop(other, jnp.subtract)

    def __rsub__(self, other):
        """Right subtraction (other - self).

        Args:
            other (Number): Left operand.

        Returns:
            ManagedArray: Result of subtraction.
        """
        return self._rop(other, jnp.subtract)

        return srec.copy_with(data=srec.data - orec.data)

    def _reduce_dims(self, dims=None):
        """Prepare dimensions for reduction operations.

        Args:
            dims (str | list[str] | None): Dimensions to reduce over.

        Returns:
            tuple: (axis_indices, remaining_dims) for the reduction.
        """
        if dims is None:
            lifted_ax = None
            reduced_dims = self.dims
        elif isinstance(dims, str):
            lifted_ax = self._dim_idx[dims]
            reduced_dims = [d for d in self.dims if d != dims]
        else:
            lifted_ax = tuple([self._dim_idx[d] for d in dims])
            reduced_dims = [d for d in self.dims if d not in dims]
        return lifted_ax, reduced_dims

    def _liftreduction(self, op, dims=None):
        """Apply a reduction operation over specified dimensions.

        Args:
            op (str): Name of the reduction operation (e.g., 'sum', 'mean').
            dims (str | list[str] | None): Dimensions to reduce over.

        Returns:
            ManagedArray | scalar: Result of the reduction.
        """
        lifted_ax, reduced_dims = self._reduce_dims(dims)
        reduced_data = getattr(self.data, op)(axis=lifted_ax)

        if len(reduced_dims) == 0:
            return reduced_data
        else:
            reduced_indexers = {
                k: indexer
                for k, indexer in self.indices.items()
                if indexer.dim in reduced_dims
            }

        return self.copy_with(
            data=reduced_data, dims=reduced_dims, indices=reduced_indexers
        )

    @property
    def shape(self):
        """Shape of the underlying data array.

        Returns:
            tuple[int, ...]: The shape tuple.
        """
        return self.data.shape

    def query(self, *args, **kwargs):
        """Query the array using named indices.

        Args:
            *args: Single positional argument for single-index arrays.
            **kwargs: Named index queries (index_name=query_spec).

        Returns:
            ManagedArray: A new array with the queried subset.

        Raises:
            ValueError: If single arg used with multi-index array.
            Exception: If both args and kwargs provided, or unsupported slice.
        """
        if len(args) == 1 and len(kwargs) == 0:
            if len(self.indices) == 1:
                idx_name = list(self.indices)[0]
            else:
                raise ValueError("Single argument requires single index")
            qargs = {idx_name: args[0]}
        elif len(args) == 0 and len(kwargs) > 0:
            qargs = kwargs
        else:
            raise Exception("Only one of args or kwargs can be supplied")

        qindices = []
        for idx_name, q in qargs.items():
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
            # +++ Slightly ugly hack to help out all the 1d compartmentarray indexing
            if len(self.dims) == 1:
                parent_indices = slicer[0]
            else:
                parent_indices = slicer
            return self.copy_with(
                data=out_data, indices=new_indices, parent_indices=parent_indices
            )
        except:
            raise Exception("Unsupported slice styles; try chaining queries")

    def __repr__(self):
        """Return string representation of the ManagedArray.

        Returns:
            str: Formatted string showing dimensions, shape, indices, and data.
        """
        return (
            f"ManagedArray\n{self.dims} {self.shape}\n"
            + f"Indices:\n{list(self.indices)}\n"
            + f"Data:\n{self.data}"
        )

    def sumcats(self, *args, **kwargs) -> ManagedArray:
        """Sum over category groups to create category aggregations.

        Args:
            *args: Category groups for single-index arrays.
            **kwargs: Named category groups (index_name=category_groups).

        Returns:
            ManagedArray: New array with categories summed and replaced
                         with a 'category' dimension.

        Raises:
            Exception: If argument format is invalid.
        """

        from .categories import get_cat_indices_list, ManagedCategoryGroupIndex

        if (len(args) > 0 and len(kwargs) > 0) or len(args) > 1 or len(kwargs) > 1:
            raise Exception("Only one positional or one kwarg allowed")
        elif len(args) == 1 and len(kwargs) == 0:
            if len(self.indices) == 1:
                idx_name = list(self.indices.keys())[0]
                catgroups = args[0]
            else:
                raise Exception(
                    "Must supply index name in kwarg for multi-index ManagedArray"
                )
        elif len(kwargs) == 1:
            idx_name, catgroups = list(kwargs.items())[0]
        else:
            raise Exception("Unmatched argument types")

        indexer = self.indices[idx_name]
        maps_dim, cat_cmap = indexer.dim, indexer.index
        cat_indices = get_cat_indices_list(catgroups, cat_cmap)
        # cat_names = get_category_names(catgroups)
        dim_idx = self._dim_idx[maps_dim]

        if len(set([len(c) for c in cat_indices])) == 1:
            # Homogenous case - can do this as one op
            slicers = [
                slice(None) if i != dim_idx else np.array(cat_indices)
                for i in range(len(self.dims))
            ]
            #
            new_data = self.data[*slicers].sum(axis=dim_idx + 1)
            out_dims = [d if d != maps_dim else "category" for d in self.dims]

        else:
            # Category slices of differing lengths, need to perform
            # as separate ops then concatenate into final array
            cat_data = []
            for c in cat_indices:
                slicers = [
                    slice(None) if i != dim_idx else np.array(c)
                    for i in range(len(self.dims))
                ]
                cat_data.append(self.data[*slicers].sum(axis=dim_idx))
            new_data = jnp.array(cat_data)
            out_dims = ["category"] + [d for d in self.dims if d != maps_dim]

        target_dims = [d if d != maps_dim else "category" for d in self.dims]

        out_indices = {
            name: midx for name, midx in self.indices.items() if midx.dim != maps_dim
        }
        out_indices["category"] = ManagedCategoryGroupIndex("category", catgroups)
        out_ma = self.copy_with(data=new_data, dims=out_dims, indices=out_indices)

        return out_ma.transpose(target_dims)

    def sum(self, dims=None, to_dims=None):
        """Sum over specified dimensions.

        Args:
            dims (str | list[str] | None): Dimensions to sum over.
            to_dims (str | list[str] | None): Dimensions to preserve
                                            (alternative to dims).

        Returns:
            ManagedArray | scalar: Result of summation. Returns scalar
                                  if all dimensions are summed.

        Raises:
            Exception: If both dims and to_dims are provided.
        """
        if to_dims is not None:
            if dims is not None:
                raise Exception("Only one of dims and to_dims can be supplied")
            if isinstance(to_dims, str):
                to_dims = [to_dims]
            dims = [d for d in self.dims if d not in to_dims]
        elif dims is None:
            return self.data.sum()
        return self._liftreduction("sum", dims=dims)

    def rolling(self, window: int, reduction_func: Callable) -> ManagedArray:
        """Perform a rolling reduction over the the first dimension, of length (window), using reduction_func
        The following would be equivalent to pd.DataFrame.rolling(7).sum()
        : ma.rolling(7, jnp.sum)

        Args:
            window: Length of window to
            reduction_func: Callable taking a single (jax) array argument

        Returns:
            ManagedArray of same indices as original, but with rolling function applied
        """
        reduced_data = get_rolling_reduction(reduction_func, window)(self.data)
        return self.copy_with(data=reduced_data)

    def to_pandas_df(self):
        """Convert to a pandas DataFrame.

        Only supports 1D and 2D arrays. For 2D arrays, uses dimension indices
        and labellers to create appropriate column names.

        Returns:
            pd.DataFrame: DataFrame representation of the array.

        Raises:
            Exception: If array has more than 2 dimensions.
        """
        if len(self.dims) > 2:
            raise Exception("Only 2d ManagedArrays supported for Pandas export")

        if len(self.dims) == 1:
            columns = ["data"]
        else:
            data_dim = self.dims[1]

            if data_dim in self.labellers:
                labeller = self.labellers[data_dim]
                columns = labeller(self)
            elif data_dim in self.indices:
                dim_indexer = self.indices[data_dim]
                if hasattr(dim_indexer, "get_labels"):
                    columns = dim_indexer.get_labels()
                else:
                    col_idx = self.indices[data_dim].index
                    if isinstance(col_idx, proto.CompartmentContainer):
                        columns = col_idx.get_labels()
                    else:
                        columns = col_idx
            else:
                columns = None

        return pd.DataFrame(
            index=self.indices["time"].index, data=self.data, columns=columns
        )

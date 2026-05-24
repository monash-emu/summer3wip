import numpy as np
from jax import Array

from .managed import ManagedArray
from .categories import CategoryData, get_cat_indices_list


def mul_jarray_catdata(data: Array, cat_data: CategoryData, cmap, unique_indices: bool = False):
    cat_indices_l = get_cat_indices_list(cat_data.cats, cmap)
    if len(set([len(ci) for ci in cat_indices_l])) != 1:
        for i, ci in enumerate(cat_indices_l):
            data = data.at[ci].mul(cat_data.data[i], unique_indices=unique_indices)
    else:
        cat_indices_arr = np.array(cat_indices_l)
        data = data.at[cat_indices_arr.T].mul(cat_data.data, unique_indices=unique_indices)
    return data


def mul_ma_catdata(arr: ManagedArray, cat_data: CategoryData, unique_indices: bool = False):
    cmap = arr.indices["compartment"].index

    data = mul_jarray_catdata(arr.data, cat_data, cmap, unique_indices)

    return arr.copy_with(data=data)

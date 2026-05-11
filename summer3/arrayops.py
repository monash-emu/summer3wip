import numpy as np

from .managed import ManagedArray
from .categories import CategoryData, get_cat_indices_list


def mul_ma_catdata(arr: ManagedArray, cat_data: CategoryData):
    cmap = arr.indices["compartment"].index
    cat_indices_l = get_cat_indices_list(cat_data.cats, cmap)
    data = arr.data
    if len(set([len(ci) for ci in cat_indices_l])) != 1:
        for i, ci in enumerate(cat_indices_l):
            data = data.at[ci].mul(cat_data.data[i])
    else:
        cat_indices_arr = np.array(cat_indices_l)
        data = data.at[cat_indices_arr.T].mul(cat_data.data)
    return arr.copy_with(data=data)

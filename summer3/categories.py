from __future__ import annotations

from typing import Iterable
from .proto import *
from .managed import *

from .utils import squash_to_slice, validate_qspec


class CategoryData(ManagedArray):
    def __init__(self, cats: "CategoryGroup", data: Array):

        dshape = data.shape
        if (ncats := len(cats)) != len(data) or len(dshape) > 1:
            raise ValueError(
                f"Shape mismatch between categories ({ncats}) and data {dshape}"
            )

        indexer = ManagedCategoryGroupIndex("category", cats)
        super().__init__(data=data, dims=["category"], indices={"category": indexer})
        self.cats = cats

    def __repr__(self):
        return f"CategoryData:\n{self.cats}\n{self.data}\n"


class Category:
    def __init__(self, traits: list[StratSpec]):
        validate_qspec(traits)
        self.traits = validate_qspec(traits)

    def __repr__(self):
        return "Category: " + repr(self.traits)

    # def __eq__(self, other):
    #    return set(self.strata) == set(other.strata)

    def __hash__(self):
        return hash(tuple(*(self.traits,)))

    def matches(self, other, traits=None):
        if isinstance(traits, Stratification):
            traits = [traits]
        a_strats = {strat: strata for (strat, strata) in self.traits}
        b_strats = {strat: strata for (strat, strata) in other.traits}
        if traits is None:
            traits = other.traits
        for strat, strata in traits:
            if set(a_strats.get(strat)) != set(b_strats.get(strat)):
                return False
        return True

    def __add__(self, other):
        return Category(self.traits + other.traits)


class CategoryGroup:
    def __init__(
        self,
        categories: list[Category],
        indices=None,
        parent=None,
        names: Optional[Iterable[str]] = None,
    ):
        self.categories = categories
        self.indices = indices or np.arange(len(categories))
        self.parent = parent or self
        if names is not None:
            self.set_names(names)
        else:
            self.names = None

    def set_names(self, names: Iterable[str]):
        names = tuple(names)
        if (len(names) != len(self.categories)) or len(set(names)) != len(names):
            raise ValueError(
                "Names and categories must map uniquely", names, self.categories
            )
        self.names = names

    def query(self, q: list[StratSpec]):
        qval = validate_qspec(q)
        valid_cats = []
        valid_indices = []
        for i, cat in enumerate(self.categories):
            if cat.matches(Category(qval)):
                valid_cats.append(cat)
                valid_indices.append(i)

        if len(valid_cats) == 0:
            raise KeyError(q)

        return CategoryGroup(valid_cats, np.array(valid_indices), self)

    def __len__(self):
        return len(self.categories)

    def __iter__(self):
        return self.categories.__iter__()

    def _product_catgroup(self, other: "CategoryGroup"):
        categories = []
        for cat in self.categories:
            for other_cat in other.categories:
                categories.append(cat + other_cat)
        return CategoryGroup(categories)

    def product(self, trait: Union[StratSpec, "CategoryGroup", Stratification]):
        if isinstance(trait, Stratification):
            trait = trait.categories()
        if isinstance(trait, CategoryGroup):
            return self._product_catgroup(trait)
        return CategoryGroup([cat + Category(trait) for cat in self.categories])

    def wrap(self, data) -> CategoryData:
        return CategoryData(self, data)

    def __repr__(self):
        return "CategoryGroup\n" + "\n".join([repr(c) for c in self.categories])

    def __getitem__(self, key) -> Union[Category, CategoryGroup]:
        if isinstance(key, int):
            return self.categories[key]
        elif isinstance(key, slice):
            return CategoryGroup(self.categories[key])
        elif isinstance(key, Iterable):
            return CategoryGroup([self.categories[subk] for subk in key])
        else:
            raise ValueError("Invalid key type", key)

    def strats(self):
        strats = []
        for cat in self.categories:
            for strat, strata in cat.traits:
                strats.append(strat)
        return list(set(strats))


class ManagedCategoryGroupIndex(ManagedIndex):
    def __init__(self, dim: str, index: CategoryGroup):
        super().__init__(dim, index)

    def query(self, q) -> tuple["ManagedCategoryGroupIndex", Indexer]:
        qres = self.index.query(q)
        return ManagedCategoryGroupIndex(self.dim, qres), squash_to_slice(qres.indices)

    def get_labels(self):
        def label_for_category(category):
            return "_".join(["|".join(strata) for strat, strata in category.traits])

        return [label_for_category(cat) for cat in self.index.categories]

    def __repr__(self):
        return f"ManagedCategoryGroupIndex: maps [{self.dim}]\n" + repr(self.index)


def category_idx_reduction(cat_indices: list[np.ndarray], src: jax.Array):
    if len(set([len(c) for c in cat_indices])) == 1:
        return src[np.array(cat_indices)].sum(axis=1)
    else:
        return jnp.array([src[c].sum() for c in cat_indices])


def query_cat_reduction(query_cats, comp_data):
    if isinstance(query_cats, CategoryGroup):
        query_cats = query_cats.categories
    indices = [comp_data.query(qc).parent_indices for qc in query_cats]
    return category_idx_reduction(indices, comp_data.data)


def get_cat_indices(query_cats, comp_data):
    if isinstance(query_cats, CategoryGroup):
        query_cats = [c.traits for c in query_cats.categories]
    indices = [comp_data.query(qc).parent_indices for qc in query_cats]
    return np.array(indices)


def strat_data_from_pandas(series, stratification):
    assert set(series.index) == set(stratification.strata)
    data = series[list(stratification.strata)].to_numpy()
    return CategoryData(stratification.categories(), data)


def cat_data_from_pandas(df, cat_group: CategoryGroup, value_col="value"):
    cgroup_strats = strats_for_cat_group(cat_group)
    data = np.zeros(len(cat_group))
    for i, cat in enumerate(cat_group.categories):
        filtered = df
        for strat, strata in cat.traits:
            assert len(strata) == 1
            filtered = filtered[filtered[strat.name] == strata[0]]
        data[i] = filtered[value_col].iloc[0]
    return CategoryData(cat_group, data)


def get_cat_indices_list(query_cats, comp_data, flat=False):
    if isinstance(query_cats, CategoryGroup):
        query_cats = [c.traits for c in query_cats.categories]
    indices = [comp_data.query(qc).parent_indices for qc in query_cats]
    return indices

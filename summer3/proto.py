from __future__ import annotations

import numpy as np
import itertools
from typing import Optional, Sequence, Union
from warnings import warn
from copy import deepcopy
import numpy.typing as npt

from .utils import validate_qspec, strats_for_cmap

import jax
from jax import numpy as jnp
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .categories import CategoryGroup
    from .managed import ManagedArray


class Stratification:
    def __init__(self, name: str, strata: list[str]):
        self.name = name
        self.strata = tuple(strata)

    def __repr__(self):
        return f"Stratification: {self.name}"

    def __getitem__(self, k):
        if isinstance(k, str):
            if k in self.strata:
                return (
                    self,
                    tuple(
                        [
                            k,
                        ]
                    ),
                )
            else:
                raise KeyError()
        elif k is Ellipsis:
            return (self, tuple([strat for strat in self.strata]))
        else:
            strata = [ki for ki in k]
            for s in strata:
                if s not in self.strata:
                    raise KeyError()
            return (self, tuple(strata))

    def categories(self, strata=None) -> CategoryGroup:
        from .categories import CategoryGroup, Category

        if strata is None:
            strata = self.strata
        return CategoryGroup([Category((self, [stratum])) for stratum in strata])

    # +++
    # Provide an easy way to obtain StratSpec - maybe getitem?


class Compartment:
    def __init__(self, strata: list[tuple[Stratification, str]]):
        self.strata = strata

    def __repr__(self):
        return "Compartment :" + repr(self.strata)

    # def __eq__(self, other):
    #    return set(self.strata) == set(other.strata)

    def __hash__(self):
        return hash(tuple(*(self.strata,)))

    def matches(self, other, strats):
        a_strats = {strat: stratum for (strat, stratum) in self.strata}
        b_strats = {strat: stratum for (strat, stratum) in other.strata}
        for strat in strats:
            if a_strats.get(strat) != b_strats.get(strat):
                return False
        return True


StratSpec = (
    tuple[Stratification, str]
    | tuple[Stratification, list[str]]
    | tuple[Stratification, tuple[str]]
    | None
)
StratMap = dict[Stratification, StratSpec]
CompartmentArray = Sequence[Compartment]


class CompartmentContainer:
    compartments: CompartmentArray

    def __init__(
        self,
        compartments: CompartmentArray,
        root: Optional["CompartmentMap"] = None,
        parent: Optional["CompartmentContainer"] = None,
        indices: Optional[np.ndarray] = None,
    ):
        self.compartments = np.array(compartments)
        if parent is None and indices is None:
            parent = self
            indices = np.arange(len(compartments))

        if parent is not None and indices is not None:
            self.parent = parent
            self.parent_indices = indices
        else:
            raise Exception("Both or neither of parent and indices must be specified")

        self.root = root

    def __getitem__(self, indices):
        compartments = self.compartments[indices]
        return CompartmentContainer(compartments, self.root, self, indices)

    def query(self, traits: list[StratSpec]) -> CompartmentContainer:
        traits = validate_qspec(traits)
        qres = []
        indices = []
        for i, c in enumerate(self.compartments):
            has_all = True
            for t in traits:
                has_trait = False
                for sspec in iter_stratspec(t):
                    has_trait = has_trait or sspec in c.strata
                if not has_trait:
                    has_all = False
                    break
            if has_all:
                qres.append(c)
                indices.append(i)
        return CompartmentContainer(np.array(qres), self.root, self, np.array(indices))

    def wrap_data(self, data):
        assert len(data) == len(
            self.compartments
        ), "Data must be of same shape as CompartmentMap"
        return CompartmentDataContainer(self.compartments, self, data)

    def zeros(self, lib=jnp):
        return CompartmentDataContainer(
            self.compartments, self, lib.zeros(len(self.compartments))
        )

    def __repr__(self):
        if self.parent == self:
            return "CompartmentContainer:\n" + repr(self.compartments)
        else:
            return (
                f"CompartmentContainer view of 0x{id(self.parent)}:\n"
                + repr(self.compartments)
                + repr(self.parent_indices)
            )

    def __len__(self):
        return len(self.compartments)

    def get_labels(self):
        def compname(c: Compartment):
            return "_".join([stratum for (strat, stratum) in c.strata])

        return [compname(c) for c in self.compartments]


class CompartmentMap(CompartmentContainer):
    def __init__(self, compartments: CompartmentArray, stratifications: StratMap):
        super().__init__(compartments)

        self.root = self
        self.stratifications = stratifications
        self._base_strat = list(stratifications)[0]
        self.remappings = {}

    @classmethod
    def new(cls, base_strat: Stratification):
        compartments = np.array(
            [Compartment([(base_strat, s)]) for s in base_strat.strata]
        )
        stratifications: dict[Stratification, Optional[tuple]] = {base_strat: None}
        return cls(compartments, stratifications)

    def stratify(
        self, strat: Stratification, stratifies: StratSpec = None, in_place=True
    ) -> Stratification:

        if stratifies is None:
            stratifies = (self._base_strat, self._base_strat.strata)
        if isinstance(stratifies[1], str):
            stratifies = (stratifies[0], [stratifies[1]])

        target_strat, target_strata = stratifies

        for existing_strat, estrat_stratifies in self.stratifications.items():
            if existing_strat.name == strat.name:
                warn(f"Existing stratification with name {strat.name}")
                # +++ Actually check for overlap, not just equivalency
                if estrat_stratifies == stratifies:
                    raise Exception(
                        "Existing stratification with same name overlaps",
                        strat.name,
                        stratifies,
                    )
        out_comps = []
        new_comps = []
        i = 0

        remapped_comps = {}

        for c in self.compartments:
            if any(
                [(target_strat, t_stratum) in c.strata for t_stratum in target_strata]
            ):
                remapped_comps[c] = []
                for stratum in strat.strata:
                    new_c = Compartment(c.strata + [(strat, stratum)])
                    out_comps.append(new_c)
                    new_comps.append(new_c)
                    remapped_comps[c].append(new_c)
                    i += 1
            else:
                out_comps.append(c)
                i += 1

        if len(new_comps) == 0:
            raise Exception("No compartments match stratification request", stratifies)

        if in_place:
            self.compartments = np.array(out_comps)
            self.stratifications[strat] = stratifies
            self.remappings[strat] = remapped_comps
            self.parent_indices = np.arange(len(self.compartments))
        else:
            stratifications = self.stratifications.copy()
            stratifications[strat] = stratifies
            new_cmap = CompartmentMap(out_comps, stratifications)
            new_cmap.remappings = self.remappings.copy()
            new_cmap.remappings[strat] = remapped_comps
            return new_cmap, strat
        return strat

    def rebase(
        self, new_base_strat: Stratification, key, in_place=False
    ) -> CompartmentMap:
        new_stratifications = {new_base_strat: None}

        for k, v in self.stratifications.items():
            if k is self._base_strat:
                new_stratifications[k] = (k, [key])
            else:
                new_stratifications[k] = v
        # self.stratifications = new_stratifications
        new_c = [
            Compartment([(new_base_strat, key)] + c.strata) for c in self.compartments
        ]

        if in_place:
            self.stratifications = new_stratifications
            self._base_strat = new_base_strat
            self.compartments = new_c
            return self
        else:
            return CompartmentMap(new_c, new_stratifications)

    def add_compartments(self, other_comps: "CompartmentMap", base_stratum: str):
        idx = len(self.compartments)
        assert not any(
            [other_s in self.stratifications for other_s in other_comps.stratifications]
        )

        if base_stratum not in self._base_strat.strata:
            raise KeyError("Stratum not found in base stratification", base_stratum)

        other_rebased = other_comps.rebase(self._base_strat, base_stratum)
        for c in other_comps.compartments:

            new_c = Compartment([(self._base_strat, base_stratum)] + c.strata, idx)
            self.compartments.append(new_c)
            idx += 1

        for k, v in other_rebased.stratifications.items():
            self.stratifications[k] = v


class CompartmentDataContainer(CompartmentContainer):
    def __init__(
        self,
        compartments: CompartmentArray,
        root: CompartmentMap,
        data: jnp.array,
        parent: "CompartmentDataContainer" = None,
        indices: np.array = None,
    ):
        super().__init__(
            compartments=compartments, root=root, parent=parent, indices=indices
        )
        self.data = data

    def query(self, traits: list[StratSpec]):
        qcomp_view = super().query(traits)
        return CompartmentDataContainer(
            qcomp_view.compartments,
            self.root,
            self.data[qcomp_view.parent_indices],
            self,
            qcomp_view.parent_indices,
        )

    def as_managed_array(self) -> ManagedArray:
        from .managed import ManagedArray, ManagedIndex

        return ManagedArray(
            self.data,
            ["compartment"],
            indices={"compartment": ManagedIndex("compartment", self)},
        )

    def __repr__(self):
        if self.parent == self:
            return (
                "CompartmentDataContainer:\n"
                + repr(self.compartments)
                + repr(self.data)
            )
        else:
            return (
                f"CompartmentDataContainer view of 0x{id(self.parent)}:\n"
                f"Compartments:\n{repr(self.compartments)}\n"
                f"Indices:\n{repr(self.parent_indices)}\n"
                f"Data:\n{repr(self.data)}\n"
            )


def iter_stratspec(sspec: StratSpec):
    strat, strata = sspec
    for stratum in strata:
        yield (strat, stratum)


def reconcile_broadcast(srcq, destq, cmap, strategy=None):
    src = cmap.query(srcq)
    dest = cmap.query(destq)
    if len(src.compartments) == len(dest.compartments):
        return src, dest, None
    else:
        if len(src) > len(dest):
            print("Gather")
            src_tmp = src
            src = dest
            dest = src_tmp
            scatter = False
        else:
            print("Scatter")
            scatter = True
        src_strats = set(strats_for_cmap(src))
        dest_strats = set(strats_for_cmap(dest))
        transition_strats = set([strat for (strat, q) in srcq])
        common_strats = src_strats.intersection(dest_strats) - transition_strats
        scatters = list(dest_strats.difference(src_strats))

        comp_idx = {c: i for i, c in enumerate(cmap.compartments)}

        if len(scatters):
            scatter_strat = scatters[0]
            out_src_comps = []
            out_dest_comps = []
            out_src_indices = []
            out_dest_indices = []
            adj = []

            for src_comp in src.compartments:
                for dest_comp in dest.compartments:
                    if src_comp.matches(dest_comp, common_strats):
                        out_src_comps.append(src_comp)
                        out_dest_comps.append(dest_comp)
                        out_src_indices.append(comp_idx[src_comp])
                        out_dest_indices.append(comp_idx[dest_comp])
                        if scatter:
                            adj.append(1.0 / len(scatter_strat.strata))

            out_src_comps = np.array(out_src_comps)
            out_dest_comps = np.array(out_dest_comps)
            out_src_indices = np.array(out_src_indices)
            out_dest_indices = np.array(out_dest_indices)

            rec_src = CompartmentContainer(
                out_src_comps, src.root, src.root, out_src_indices
            )
            rec_dest = CompartmentContainer(
                out_dest_comps, dest.root, dest.root, out_dest_indices
            )
            if scatter:
                return rec_src, rec_dest, np.array(adj)
            else:
                return rec_dest, rec_src, None


class ActualizedTransitionFlow:
    def __init__(self, flow, src_cmap, dest_cmap, adjustments, apply_func):
        self.flow = flow
        self.src_cmap = src_cmap
        self.dest_cmap = dest_cmap
        self.adjustments = adjustments
        self.get_flow_vals = apply_func


class TransitionFlow:
    def __init__(self, name, srcq, destq, param):
        self.srcq = validate_qspec(srcq)
        self.destq = validate_qspec(destq)
        self.param = param
        self.adjustments = []
        self.name = name

    def actualize(self, cmap, param_key=None, adj_param_keys=None):

        # param_key = param_key or self.param
        adj_param_keys = adj_param_keys or {}

        realised_adjustments = []

        src_cmap, dest_cmap, adj = reconcile_broadcast(self.srcq, self.destq, cmap)

        if adj is not None:
            realised_adjustments.append(adj)

        from .categories import CategoryData, get_cat_indices

        def apply_flow(cdatamap, params):
            src_comp_vals = cdatamap.data[src_cmap.parent_indices]
            if param_key is None:
                param = self.param
            else:
                param = params[param_key]
            if isinstance(param, CategoryData):
                cidx = get_cat_indices(param.cats, src_cmap)

                # Need to guarantee unique indices for gradients to work
                # For most use cases this should probably be fine - where it's not we may need to split this out into
                # multiple ops, which we already do in other special cases
                assert cidx.size == np.unique(cidx).size
                flow_vals = src_comp_vals.at[cidx.T].mul(
                    param.data, unique_indices=True
                )
            else:
                flow_vals = param * src_comp_vals
            for adj in realised_adjustments:
                flow_vals = flow_vals * adj
            for i, adj in enumerate(self.adjustments):
                if i in adj_param_keys:
                    adj = params[adj_param_keys[i]]

                from .managed import ManagedArray

                if isinstance(adj, ManagedArray):
                    cats = adj.indices["category"].index
                    cidx = get_cat_indices(cats, src_cmap)
                    flow_vals = flow_vals.at[cidx.T].mul(adj.data)
                else:
                    raise Exception("Unsupported adjustment", adj)

            return flow_vals

        return ActualizedTransitionFlow(
            self, src_cmap, dest_cmap, realised_adjustments, apply_flow
        )


class ActualizedExitFlow:
    def __init__(self, flow, src_cmap, adjustments, apply_func):
        self.flow = flow
        self.src_cmap = src_cmap
        self.adjustments = adjustments
        self.get_flow_vals = apply_func


class ExitFlow:
    def __init__(self, name, srcq, param):
        self.srcq = validate_qspec(srcq)
        self.param = param
        self.adjustments = []
        self.name = name

    def actualize(self, cmap, param_key=None, adj_param_keys=None):
        src_cmap = cmap.query(self.srcq)

        adj_param_keys = adj_param_keys or {}

        from .categories import CategoryData, get_cat_indices

        def apply_flow(cdatamap, params):
            src_comp_vals = cdatamap.data[src_cmap.parent_indices]

            if param_key is None:
                param = self.param
            else:
                param = params[param_key]
            if isinstance(param, CategoryData):
                cidx = get_cat_indices(param.cats, src_cmap)

                # Need to guarantee unique indices for gradients to work
                # For most use cases this should probably be fine - where it's not we may need to split this out into
                # multiple ops, which we already do in other special cases
                assert cidx.size == np.unique(cidx).size
                flow_vals = src_comp_vals.at[cidx.T].mul(
                    param.data, unique_indices=True
                )
            else:
                flow_vals = param * src_comp_vals

            for i, adj in enumerate(self.adjustments):
                if i in adj_param_keys:
                    adj = params[adj_param_keys[i]]

                from .managed import ManagedArray

                if isinstance(adj, ManagedArray):
                    cats = adj.indices["category"].index
                    cidx = get_cat_indices(cats, src_cmap)
                    flow_vals = flow_vals.at[cidx.T].mul(adj.data)
                else:
                    raise Exception("Unsupported adjustment", adj)

            return flow_vals

        return ActualizedExitFlow(self, src_cmap, self.adjustments, apply_flow)


class ActualizedEntryFlow:
    def __init__(self, flow, dest_cmap, adjustments, apply_func):
        self.flow = flow
        self.dest_cmap = dest_cmap
        self.adjustments = adjustments
        self.get_flow_vals = apply_func


class EntryFlow:
    def __init__(self, destq, param):
        self.destq = validate_qspec(destq)
        self.param = param
        self.adjustments = []

    def actualize(self, cmap):
        dest_cmap = cmap.query(self.destq)

        def apply_flow(cdatamap, params):
            param = params[self.param]
            if isinstance(param, CategoryData):
                raise Exception("CategoryData not yet supported for EntryFlow")
            else:
                flow_vals = param
            return flow_vals

        return ActualizedEntryFlow(self, dest_cmap, self.adjustments, apply_flow)

from typing import Optional

import datetime as dt
from .runners import *
from .utils import TimeIndex, dti_to_epoch, strats_for_cmap


class CompartmentalEpiModel:
    def __init__(self, cmap: CompartmentMap, times: TimeIndex):
        self.cmap = cmap
        self.times = times
        self.flows = {}

    def set_initial_population(self, base_pops, pop_splits=None):
        self.base_pops = base_pops
        self.pop_splits = pop_splits

    def run(self, params: dict[str, float]):
        istate = build_istate(self.cmap, self.base_pops, self.pop_splits)
        cmodel = CompartmentalModelODE(self.cmap, self.flows)
        runner = cmodel.get_runner(len(self.times), dti_to_epoch(self.times))
        return runner.run(istate.data, params)

    def add_flow(self, flow, key: Optional[str] = None):
        key = key or flow.name
        if key in self.flows:
            raise KeyError("Only one flow of each name currently supported", key)
        self.flows[key] = flow


def mixing_matrix(data, source_cats: CategoryGroup, dest_cats: CategoryGroup):
    mm = ManagedArray(data, ["dest", "source"])
    mm.indices["dest"] = ManagedCategoryGroupIndex("dest", dest_cats)
    mm.indices["source"] = ManagedCategoryGroupIndex("source", source_cats)

    return mm


class InfectionProcess:
    def __init__(
        self,
        infectee_cats: CategoryGroup,
        infector_cats: CategoryGroup,
        infectious_compartments: StratSpec,
        mm: Optional[ManagedArray] = None,
    ):
        if mm is None:
            mm = mixing_matrix(
                np.ones((len(infectee_cats), len(infector_cats))),
                infector_cats,
                infectee_cats,
            )
        self.mm = mm
        self.infector_cats = infector_cats
        self.infectee_cats = infectee_cats
        self.infectious_compartments = infectious_compartments
        self._infectious_pop_cats = self.infector_cats.product(infectious_compartments)

    def process(self, compartment_values: ManagedArray, contact_rate: float):
        ipops = compartment_values.sumcats(self._infectious_pop_cats)
        total_pop = compartment_values.sumcats(self.infector_cats)
        age_foi = (self.mm.data @ (ipops.data / total_pop.data)) * contact_rate
        return CategoryData(self.infectee_cats, age_foi)


def build_istate(cmap, cat_data: CategoryData, pop_splits=None):
    pop_splits = pop_splits or []
    istate = cmap.zeros().as_managed_array()
    all_strats = set(strats_for_cmap(cmap))
    cat_strats = set(cat_data.cats.strats())
    strats_to_split = all_strats.difference(cat_strats)

    cat_indices_l = get_cat_indices_list(cat_data.cats, cmap)
    cat_indices = np.array(cat_indices_l)

    data = istate.data.at[cat_indices.T].set(cat_data.data)

    for ps_cat_data in pop_splits:
        this_pop_strats = set(ps_cat_data.cats.strats())
        strats_to_split = strats_to_split.difference(this_pop_strats)
        cat_indices_l = get_cat_indices_list(ps_cat_data.cats, cmap)
        if len(set([len(ci) for ci in cat_indices_l])) > 1:
            # We need to iterate the data updates if the cat indices are of different lengths, otherwise we can do it in one go
            for cat_local_idx, cat_idx in enumerate(cat_indices_l):
                data = data.at[cat_idx].mul(ps_cat_data.data[cat_local_idx])
        else:
            cat_indices = np.array(cat_indices_l)
            data = data.at[cat_indices.T].mul(ps_cat_data.data)

    for rem_strat in list(strats_to_split):
        cgroup = rem_strat.categories()
        cat_indices = np.array(get_cat_indices_list(cgroup, cmap))
        data = data.at[cat_indices.T].mul(1.0 / len(rem_strat.strata))

    return istate.copy_with(data=data)

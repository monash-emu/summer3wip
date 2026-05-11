import pandas as pd
import numpy as np
import datetime as dt

from summer3.graph import *
from summer3.epi import *


def test_compartment_map_from_strat():
    disease_state = Stratification("disease_state", ["S", "I", "R"])
    humans = CompartmentMap.new(disease_state)
    clist = list(c.strata[0][1] for c in humans.compartments)
    assert clist == ["S", "I", "R"]

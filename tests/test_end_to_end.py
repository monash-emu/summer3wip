import pandas as pd
import numpy as np
import datetime as dt
import jax.numpy as jnp

from summer3.graph import *
from summer3.epi import *
from summer3.categories import CategoryData, strat_data_from_pandas
from summer3.utils import dti_to_epoch


def test_compartment_map_from_strat():
    disease_state = Stratification("disease_state", ["S", "I", "R"])
    humans = CompartmentMap.new(disease_state)
    clist = list(c.strata[0][1] for c in humans.compartments)
    assert clist == ["S", "I", "R"]


def test_end_to_end_sir_with_age_stratification():
    """
    Full end-to-end test of a stratified SIR model with age structure,
    based on notebooks/01-Example.ipynb.

    Model features:
    - Disease stratification: S, I, R compartments
    - Age stratification: child, adult
    - Infection process with force of infection
    - Recovery flow
    - Birth and death flows
    - Heterogeneous initial population distribution
    """

    # ========== 1. Set up compartmental structure ==========
    disease_state = Stratification("disease_state", ["S", "I", "R"])
    humans = CompartmentMap.new(disease_state)

    # Add age stratification
    age_strat = humans.stratify(Stratification("age", ["child", "adult"]))

    # ========== 2. Define infection process ==========
    age_cats = age_strat.categories()
    infectees = age_cats
    infectors = age_cats

    # Create infection process with homogeneous mixing
    iprocess = defer(InfectionProcess)(infectees, infectors, disease_state["I"])

    # Force of infection using contact rate parameter
    foi = defer(InfectionProcess.process)(
        iprocess, CompartmentValues, Parameter("contact_rate", 0.2)
    )

    # ========== 3. Define flows ==========
    # Disease flows
    infection = TransitionFlow("infection", disease_state["S"], disease_state["I"], foi)
    recovery = TransitionFlow(
        "recovery",
        disease_state["I"],
        disease_state["R"],
        1.0 / Parameter("recovery_time", 10.0),
    )

    # Demographic flows
    birth = EntryFlow("birth", age_strat["child"], Parameter("birth_rate", 0.01))
    death = ExitFlow(
        "death", age_strat["child", "adult"], Parameter("death_rate", 0.01)
    )

    # ========== 4. Construct model ==========
    # Convert to list of datetime objects for TimeIndex type
    times = list(pd.date_range("2020-01-01", "2020-12-31", freq="D"))
    epi_model = CompartmentalEpiModel(humans, times)

    epi_model.add_flow(infection)
    epi_model.add_flow(recovery)
    epi_model.add_flow(birth)
    epi_model.add_flow(death)

    # ========== 5. Set initial populations ==========
    # Initial population by age group
    pop_data = pd.Series(index=["child", "adult"], data=np.array([1000.0, 1500.0]))
    base_pops = strat_data_from_pandas(pop_data, age_strat)

    # Initial disease distribution: 90% S, 10% I, 0% R (same for all age/location)
    pop_splits = [CategoryData(disease_state.categories(), jnp.array([0.9, 0.1, 0.0]))]

    epi_model.set_initial_population(base_pops, pop_splits)

    # ========== 6. Run model ==========
    params = {
        "contact_rate": 0.1,
        "recovery_time": 50.0,
        "birth_rate": 5.0,
        "death_rate": 0.1,
    }

    results = epi_model.run(params)

    # ========== 7. Verify results structure ==========
    # Check that we got all expected result components
    assert "compartments" in results
    assert "flows" in results
    assert "computed_values" in results

    compartments = results["compartments"]
    flows = results["flows"]

    # ========== 8. Validate compartment outputs ==========
    # Should have time and compartment dimensions
    assert compartments.dims == ["time", "compartment"]
    assert len(compartments.dims) == 2

    # Time dimension should match input times
    assert len(compartments.data) == len(times)

    # Compartment dimension should be total compartments (3 disease states × 2 age groups)
    assert compartments.data.shape[1] == 6

    # All compartment values should be non-negative
    assert jnp.all(compartments.data >= 0.0)

    # Total population should not explode (sanity check)
    total_pop_t0 = jnp.sum(compartments.data[0, :])
    total_pop_tf = jnp.sum(compartments.data[-1, :])
    assert total_pop_t0 > 0.0
    assert total_pop_tf > 0.0
    assert total_pop_tf < total_pop_t0 * 10.0  # Allow for growth but not explosion

    # ========== 9. Validate flow outputs ==========
    # Check that all flows are present and have correct dimensions
    expected_flows = {"infection", "recovery", "birth", "death"}
    assert set(flows.keys()) == expected_flows

    for flow_name, flow_data in flows.items():
        # Each flow should have time dimension
        assert "time" in flow_data.dims
        assert len(flow_data.data) == len(times)
        # Flow values should be non-negative
        assert jnp.all(flow_data.data >= 0.0)

    # ========== 10. Validate infection dynamics ==========
    # Infection flow should decrease over time (epidemic burn-out)
    infection_flow = flows["infection"]
    infection_by_time = jnp.sum(infection_flow.data, axis=1)

    # Peak should occur before end of simulation
    peak_idx = jnp.argmax(infection_by_time)
    assert 0 < peak_idx < len(times) - 1

    # Recovery flow should be positive throughout
    recovery_flow = flows["recovery"]
    recovery_by_time = jnp.sum(recovery_flow.data, axis=1)
    assert jnp.all(recovery_by_time >= 0.0)

    # ========== 11. Validate recovered compartment growth ==========
    # Query recovered compartments and verify they increase over time
    # Get indices of R compartments
    recovered_indices = [
        i
        for i, c in enumerate(humans.compartments)
        if any(s[1] == "R" for s in c.strata)
    ]

    recovered_t0 = jnp.sum(compartments.data[0, recovered_indices])
    recovered_tf = jnp.sum(compartments.data[-1, recovered_indices])

    # Recovered should increase (epidemic occurred)
    assert recovered_tf > recovered_t0

    # ========== 12. Integration test: results queryability ==========
    # Verify that ManagedArray query operations work
    comp_summed = compartments.sumcats(compartment=age_cats)
    assert len(comp_summed.data) == len(times)
    assert comp_summed.data.shape[1] == len(age_cats)  # 3 disease states

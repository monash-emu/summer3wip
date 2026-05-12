# Summer3 AI Coding Agent Guide

## Project Overview

**Summer3** is a JAX-based epidemiological modeling framework that uses a declarative, compute-graph approach to define disease dynamics. It's the spiritual successor to Summer2, redesigned for performance and flexibility using functional programming patterns.

### Key Differentiators
- **JAX-first**: All computations use JAX arrays for JIT compilation and automatic differentiation
- **Compute Graph DSL**: Models are defined declaratively as computation graphs (not imperative loops)
- **Categorical Stratification**: Multi-dimensional population structures via categories/stratifications (age, location, disease state, etc.)
- **Deferred Evaluation**: The `defer()` pattern wraps functions into graph nodes without immediate execution

---

## Architecture: Five Core Subsystems

### 1. **Compartmental Model Definition** (`proto.py`, `categories.py`)
- **Stratification**: Named dimensions for model compartments (e.g., "disease_state" with ["S","I","R"])
- **Compartment**: A tuple of (stratification, stratum) pairs defining a state (e.g., disease_state["S"])
- **CompartmentMap**: Container of all compartments; manages indexing and parent-child relationships
- **Category/CategoryGroup**: Represent subsets of compartments for querying and operations
- **Key Pattern**: Use bracket notation: `disease_state["I"]` or `age_strat["child","adult"]` for querying

### 2. **Managed Arrays & Indices** (`managed.py`)
Central abstraction for labeled multi-dimensional arrays with semantic dimension names.

**Key Classes:**
- `ManagedArray`: Wraps JAX arrays with named dimensions and indices
  - `.data`: The actual JAX array
  - `.dims`: List of dimension names ["time", "compartment"]
  - `.indices`: Dict mapping dim names to `ManagedIndex` objects
- `ManagedIndex`: Maps dimension names to index data (pd.Index, CompartmentContainer, etc.)
- Operations: `.query()`, `.sumcats()`, `.as_managed_array()`

**Pattern**: Operations preserve dimension semantics through indices; queries return subsets with proper indexing.

### 3. **Computation Graph Engine** (`computegraph/`)
Transforms declarative graph specifications into JAX-compatible callables.

**Key Components:**
- `Variable`: Base for parameters and model variables (`Parameter`, `ModelVariable`)
- `Function`: Deferred function call (created by `defer()` wrapper)
- `ComputeGraph`: Builds a NetworkX DAG from graph_dict, traces dependencies, compiles to callable
- **Execution**: `cgraph.get_callable()` returns a function accepting `model_variables` and `parameters` dicts

**Key Pattern**: Use `defer(func)(args)` to build graph nodes instead of calling functions directly.

```python
# DON'T: foi = InfectionProcess.process(iprocess, CompartmentValues, Parameter("contact_rate", 0.2))
# DO: foi = defer(InfectionProcess.process)(iprocess, CompartmentValues, Parameter("contact_rate", 0.2))
```

### 4. **Epidemiological Flows** (`runners.py`, `epi.py`)
Models transition between compartments using flow functions.

**Flow Types:**
- `TransitionFlow(name, source_comp, dest_comp, rate_or_function)`: Between compartments
- `EntryFlow(name, dest_comp, rate_function)`: Birth/immigration
- `ExitFlow(name, source_comp, rate_function)`: Death/emigration

**Key Pattern**: Flows are added to `CompartmentalEpiModel` via `.add_flow()`. Rate functions can be scalars, Parameters, or deferred Functions (including `InfectionProcess.process` output).

### 5. **Model Runners & Execution** (`runners.py`, `engine.py`)
Two execution paths with different performance characteristics:

- **NaiveModel**: Direct state update loops (fast development)
- **GraphModel**: Compute-graph-based (enables differentiation, optimization)
- **CompartmentalModelODE**: Wrapper for ODE solvers (diffrax integration)

Both use `lax.scan()` for timestep iteration; can be JIT-compiled.

---

## Critical Data Flows

### Model Initialization → Execution → Results

```
CompartmentalEpiModel (structure)
  ├─ CompartmentMap (cmap) - compartments & stratifications
  ├─ flows: dict[str, Flow] - transition definitions
  └─ times: TimeIndex - time points

              ↓ .run(params)

CompartmentalModelODE (actualize flows)
  ├─ actual_flows - flows compiled to JAX operations
  └─ get_runner(timesteps, epochs, jit=True)

              ↓ runner.run(init_state, params)

CompartmentalModelRunner (execution)
  ├─ ManagedArray results for compartments
  ├─ flow outputs dict[flow_name → ManagedArray]
  └─ computed_values dict[graph_node_name → ManagedArray]
```

### Building Initial State

```
pop_data (pd.Series, indexed by category names)
  ↓ strat_data_from_pandas()
base_pops (CategoryData)

pop_splits (list[CategoryData]) - optional fractional distributions
  ↓ build_istate()
istate (ManagedArray) - JAX array with compartment indexing
```

---

## Project-Specific Patterns & Conventions

### 1. Query Specifications (StratSpec)
Used throughout to select compartments/categories:
```python
disease_state["I"]                  # Single stratum
age_strat["child", "adult"]        # Multiple strata
disease_state[...]                 # All strata (ellipsis)
[("disease_state", ["I"]), ...]   # Tuples: (Stratification, strata_list)
```

### 2. Deferred Execution in Graphs
Always wrap flow computations in `defer()`:
```python
# Mixing matrix effect on infection
iprocess = defer(InfectionProcess)(infectees, infectors, disease_state["I"])
foi = defer(InfectionProcess.process)(iprocess, CompartmentValues, Parameter("contact_rate", 0.2))
infection_flow = TransitionFlow("infection", disease_state["S"], disease_state["I"], foi)
```

### 3. Testing & Notebooks
- Tests use pytest (files in `tests/`)
- Examples in `notebooks/01-Example.ipynb` show full workflow (stratification → model definition → execution)
- Scratch notebooks for experimentation in `notebooks/scratch.ipynb`

### 4. Dimension & Index Naming Conventions
- Dimensions: lowercase with underscores ("compartment", "time", "source", "dest")
- Indices: `ManagedIndex(dim_name, index_data)` - third arg is pd.Index, CompartmentContainer, or CategoryGroup
- Results: Always wrap time-series results in ManagedArray with explicit dims/indices

### 5. JAX Integration Points
- **Arrays**: All numerical data must be JAX arrays or convertible to jnp
- **JIT Compilation**: Use `jit=True` flag in runner creation; `lax.scan()` for loops, `jnp.clip()` for bounds
- **Automatic Differentiation**: Enabled by keeping compute graph traceable (no control flow on compartment values)

---

## Development Workflows

### Environment Setup
**Always use the Pixi environment** for all development, testing, and benchmarking work. Pixi ensures reproducible builds and dependency isolation.

### Running Tests
```bash
# Always use pixi for testing
pixi run pytest tests/

# With coverage reporting
pixi run pytest tests/ --cov=summer3

# Watch mode for development
pixi run pytest tests/ -v --tb=short
```

### Running Notebooks
```bash
# Start Jupyter in the pixi environment
pixi run jupyter lab notebooks/
```

### Benchmarking & Profiling
```bash
# Profile model execution within pixi
pixi run python -m cProfile -o profile.stats your_script.py
pixi run python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

### Exploring in Notebooks
- Start from `notebooks/01-Example.ipynb`
- Import pattern: `from summer3.epi import *; from summer3.graph import *`
- Always build models step-by-step (compartment → stratification → flows → run)
- Run all notebooks within the pixi environment for consistency

### Adding New Features
1. **New Flow Type**: Extend `Flow` base class in runners.py; implement `actualize()` and `get_flow_vals()`
2. **New Compartment Query**: Add methods to `CategoryGroup` or extend `query()` logic in categories.py
3. **Custom Compute Operations**: Wrap in `defer()` and add to graph_dict before creating ComputeGraph
4. **Model Outputs**: Extend result wrapping in `CompartmentalModelRunner.run()` to handle new computed values

### Common Imports by Use Case
```python
# Model structure
from summer3.graph import Stratification, CompartmentMap, CompartmentValues
from summer3.proto import CompartmentMap, Compartment, Stratification

# Building flows
from summer3.epi import CompartmentalEpiModel, TransitionFlow, EntryFlow, ExitFlow, InfectionProcess
from summer3.graph import Parameter, defer

# Data handling
from summer3.managed import ManagedArray, ManagedIndex, ManagedCategoryGroupIndex
from summer3.categories import CategoryData, CategoryGroup

# Execution
from summer3.runners import CompartmentalModelODE, build_istate
```

---

## Key File Reference

| File | Purpose | Entry Points |
|------|---------|--------------|
| `proto.py` | Compartments, stratifications, basic structures | `Stratification`, `Compartment`, `CompartmentMap` |
| `graph.py` | Compute graph helpers; Parameter/defer wrapping | `Parameter`, `defer()`, `label()` |
| `epi.py` | High-level epidemiological model API | `CompartmentalEpiModel`, `InfectionProcess`, `mixing_matrix()` |
| `runners.py` | Model execution engines & result wrappers | `CompartmentalModelRunner`, `build_istate()` |
| `managed.py` | Labeled arrays & indexed dimension handling | `ManagedArray`, `ManagedIndex` |
| `categories.py` | Category/stratification query & aggregation | `CategoryData`, `CategoryGroup` |
| `computegraph/` | DAG building, JIT compilation, symbolic math | `ComputeGraph`, `Variable`, `Function` |
| `computegraph/draw/` | Visualization helpers for compute graphs | `draw_compute_graph()` |

---

## Gotchas & Anti-Patterns

❌ **DON'T**: Call functions directly in flow definitions; use `defer()` to defer evaluation  
✅ **DO**: `foi = defer(InfectionProcess.process)(iprocess, CompartmentValues, Parameter(...))` 

❌ **DON'T**: Assume parameter values are available at graph-building time  
✅ **DO**: Wrap in `Parameter(key, default)` and pass via params dict at runtime

❌ **DON'T**: Mix numpy arrays with JAX arrays in flow calculations  
✅ **DO**: Ensure all flow functions operate on `ManagedArray.data` (JAX arrays)

❌ **DON'T**: Hardcode categorical indices; use `.query()` and stratification queries  
✅ **DO**: Use `disease_state["I"]` or `age_strat["child", "adult"]` for semantic clarity

---

## Dependencies & Environment

**Core** (from `pyproject.toml`):
- `jax`, `jaxlib`: Numerical computation & JIT
- `diffrax`: ODE solvers
- `pandas`, `polars`: Data handling
- `networkx`: DAG construction
- `bidict`: Bidirectional dictionary (for stratification mappings)

**Dev**: pytest (for testing)  
**Environment**: Python ≥3.10; **use Pixi** (`pixi/pixi.toml`) for reproducible builds and development

**Pixi Setup** (first-time):
```bash
# Install pixi: https://pixi.sh
pixi install  # Install dependencies from pixi.toml
pixi run pytest tests/  # Verify environment works
```

from typing import Optional
from .proto import *
from typing import NamedTuple
from .managed import ManagedArray, ManagedIndex
from .categories import *
from jax import lax, jit, grad, make_jaxpr
import jax
from .utils import Epoch
import pandas as pd
import diffrax as dfx

from summer3.computegraph import ComputeGraph
from summer3.computegraph.types import GraphObject


class SplitGraphs(NamedTuple):
    constant: ComputeGraph
    init: ComputeGraph
    timestep: ComputeGraph


def transition_flow_labeller(flowres: ManagedArray):
    if "source" in flowres.indices and "dest" in flowres.indices:
        return [
            "->".join([src, dest])
            for src, dest in zip(
                flowres.indices["source"].index.get_labels(),
                flowres.indices["dest"].index.get_labels(),
            )
        ]
    elif "source" in flowres.indices:
        return flowres.indices["source"].index.get_labels()
    elif "dest" in flowres.indices:
        return flowres.indices["dest"].index.get_labels()
    else:
        raise Exception("No valid indices found in flow result")


class CompartmentalModelODERunner:
    def __init__(
        self,
        model: "CompartmentalModelODE",
        graphs: SplitGraphs,
        actual_flows,
        run_func,
        timesteps,
        epoch: Optional[Epoch] = None,
        funcs=None,
    ):
        self.model = model

        self.graphs = graphs
        self.actual_flows = actual_flows
        self._run_func = run_func
        self.timesteps = timesteps
        self.funcs = funcs

        if epoch is None:
            self._time_idx = pd.Index(np.arange(0, timesteps))
        else:
            self._time_idx = epoch.index_to_dti(np.arange(0, timesteps))

    def run(self, init_state, params, solver_kwargs=None):
        solver_kwargs = solver_kwargs or {}
        gathered_res = self._run_func(init_state, params, solver_kwargs)
        flow_outputs = gathered_res["flows"]
        compartment_outputs = gathered_res["compartments"]
        computed_values = gathered_res["computed_values"]
        graph_res = gathered_res["graph"]

        cmap = self.model.cmap

        cdata = ManagedArray(
            compartment_outputs,
            dims=["time", "compartment"],
            indices={
                "time": ManagedIndex("time", self._time_idx),
                "compartment": ManagedIndex("compartment", cmap),
            },
        )

        flow_data = {}

        for flow_key, data in flow_outputs.items():
            actual_flow = self.actual_flows[flow_key]
            indices = {"time": ManagedIndex("time", self._time_idx)}
            if hasattr(actual_flow, "src_cmap"):
                indices["source"] = ManagedIndex("compartment", actual_flow.src_cmap)
            if hasattr(actual_flow, "dest_cmap"):
                indices["dest"] = ManagedIndex("compartment", actual_flow.dest_cmap)
            flow_data[flow_key] = ManagedArray(
                data,
                dims=["time", "compartment"],
                indices=indices,
                labellers={"compartment": transition_flow_labeller},
            )

        # Run a single timestep to get the realised coords of all items in the graph
        # Hopefully some of this disappears in optimization?
        ref_comp_outs = compartment_outputs[0, :]
        cvals_ref = self.model.cmap.zeros().as_managed_array()
        model_variables = {"time": 0.0, "compartment_values": cvals_ref}
        dyn_vals = self.graphs.timestep.get_callable()(
            model_variables=model_variables,
            parameters=params,
            static_inputs=graph_res["init"],
        )

        out_cv = {}

        for k, v in computed_values.items():
            ref_ma = dyn_vals[k]
            indices = {"time": ManagedIndex("time", self._time_idx)}
            if isinstance(ref_ma, ManagedArray):
                extra_dims = ref_ma.dims
                indices = indices | ref_ma.indices
            else:
                extra_dims = [str(i) for i in range(len(ref_ma.shape))]
            out_cv[k] = ManagedArray(v, dims=["time"] + extra_dims, indices=indices)

        return {
            "compartments": cdata,
            "flows": flow_data,
            "computed_values": out_cv,
            "aux": gathered_res.get("aux"),
            "graph": gathered_res.get("graph"),
        }


class CompartmentalModelODE:
    def __init__(self, cmap: CompartmentMap, flows: dict[str, TransitionFlow]):
        self.cmap = cmap
        self.flows = flows
        self._actual_flows = {}

    def actualize_flows(self):
        graph_dict = {}
        actual_flows = {}
        for k, flow in self.flows.items():
            if isinstance(flow.param, GraphObject):
                flow_param_key = f"_flow_param_{k}"
                graph_dict[flow_param_key] = flow.param
            else:
                flow_param_key = None
            adj_param_keys = {}
            if isinstance(flow, TransitionFlow):
                for iadj, adj in enumerate(flow.adjustments_source):
                    adj_key = f"_flow_param_{k}_adj_source[{iadj}]"
                    adj_param_keys[f"source_adj_{iadj}"] = adj_key
                    graph_dict[adj_key] = adj
                for iadj, adj in enumerate(flow.adjustments_dest):
                    adj_key = f"_flow_param_{k}_adj_dest[{iadj}]"
                    adj_param_keys[f"dest_adj_{iadj}"] = adj_key
                    graph_dict[adj_key] = adj
            else:
                for iadj, adj in enumerate(flow.adjustments):
                    if isinstance(adj, GraphObject):
                        adj_key = f"_flow_param_{k}_adj[{iadj}]"
                        adj_param_keys[iadj] = adj_key
                        graph_dict[adj_key] = adj
            actual_flows[k] = flow.actualize(self.cmap, flow_param_key, adj_param_keys)

        return ComputeGraph(graph_dict), actual_flows

    def get_runner(
        self,
        timesteps,
        epoch=None,
        jit=False,
        computed_values=None,
        default_params=None,
        dyn_params=None,
    ) -> CompartmentalModelODERunner:
        cgraph, actual_flows = self.actualize_flows()
        graphs = get_split_graphs(cgraph, dyn_params)
        # We have 3 graphs: constant, init, and timestep
        # The constant graph can be computed now

        constant_graph_res = graphs.constant.get_callable()(parameters=default_params)
        init_graph_func = graphs.init.get_callable()
        ts_graph_func = graphs.timestep.get_callable()

        # cgraphfunc = cgraph.get_callable(output_all=True)
        computed_values = computed_values or []

        # Construct a function that we can use later to get (only) the flow values
        # This is in contrast to vector_field, which is used to compute the flow values
        # accumulated into their state deltas for the ODE solver
        def get_flow_values(t, y, args):
            """
            Get the instantaneous flow values for the given time (t) and state (y)
            and parameters (params).
            """
            # Get a ManagedArray of all state ('CompartmentValues')
            params, init_graph_res = args

            cvals = self.cmap.wrap_data(y).as_managed_array()
            model_variables = {"time": t, "compartment_values": cvals}
            dyn_values = ts_graph_func(
                model_variables=model_variables,
                parameters=params,
                static_inputs=init_graph_res,
            )

            stored_flows = {}
            for k, flow in actual_flows.items():
                flow_vals = flow.get_flow_vals(cvals, dyn_values)
                stored_flows[k] = flow_vals

            return stored_flows

        # The term argument for the ODE solver
        def vector_field(t, state, args):
            params, init_graph_res = args
            cvals = self.cmap.wrap_data(state).as_managed_array()
            model_variables = {"time": t, "compartment_values": cvals}
            dyn_values = ts_graph_func(
                model_variables=model_variables,
                parameters=params,
                static_inputs=init_graph_res,
            )

            comp_delta = jnp.zeros_like(state)
            for k, flow in actual_flows.items():
                flow_vals = flow.get_flow_vals(cvals, dyn_values)
                if hasattr(flow, "src_cmap"):
                    comp_delta = comp_delta.at[flow.src_cmap.parent_indices].subtract(
                        flow_vals
                    )
                if hasattr(flow, "dest_cmap"):
                    comp_delta = comp_delta.at[flow.dest_cmap.parent_indices].add(
                        flow_vals
                    )
            return comp_delta

        def run_model(init_state, params, solver_kwargs=None, dtmax=1.0):

            solver_kwargs = solver_kwargs or {}

            term = dfx.ODETerm(vector_field)
            solver = dfx.Dopri5()
            saveat = dfx.SaveAt(ts=jnp.arange(timesteps))
            stepsize_controller = dfx.PIDController(
                rtol=1e-5, atol=1e-5, dtmax=dtmax
            )  # , dtmax=1.0)

            adjoint = dfx.RecursiveCheckpointAdjoint()
            # adjoint = diffrax.ForwardMode()
            # adjoint = diffrax.DirectAdjoint()

            init_graph_res = graphs.init.get_callable()(
                parameters=params, static_inputs=constant_graph_res
            )

            default_kwargs = dict(
                terms=term,
                solver=solver,
                t0=0,
                t1=timesteps,
                throw=False,
                max_steps=int(4 * timesteps),
                dt0=0.1,
                y0=init_state,
                args=(params, init_graph_res),
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                adjoint=adjoint,
            )

            final_kwargs = default_kwargs | solver_kwargs
            sol = dfx.diffeqsolve(**final_kwargs)

            flow_values = jax.vmap(get_flow_values, in_axes=(0, 0, None))(
                sol.ts, sol.ys, (params, init_graph_res)
            )

            def pure_jax_graph_func(t: float, cvals: jax.Array) -> dict[str, jax.Array]:
                hdata = self.cmap.wrap_data(cvals).as_managed_array()
                model_variables = {"time": t, "compartment_values": hdata}
                dyn_values = ts_graph_func(
                    model_variables=model_variables,
                    parameters=params,
                    static_inputs=init_graph_res,
                )
                return {k: dyn_values[k] for k in computed_values}

            computed_values_res = jax.vmap(pure_jax_graph_func, in_axes=(0, 0))(
                sol.ts, sol.ys
            )

            return {
                "compartments": sol.ys,
                "flows": flow_values,
                "computed_values": computed_values_res,
                "aux": sol,
                "graph": {"constant": constant_graph_res, "init": init_graph_res},
            }

            # stored_dyn = {}
            # for k in computed_values:
            #    if isinstance(dyn_values[k], ManagedArray):
            #        stored_dyn[k] = dyn_values[k].data
            #    else:
            #        stored_dyn[k] = dyn_values[k]

            # return tstep_data, {
            #    "compartments": tstep_data,
            #    "flows": stored_flows,
            #    "computed_values": stored_dyn,
            # }

        if jit:
            run_model = jax.jit(run_model)

        return CompartmentalModelODERunner(
            self,
            graphs,
            actual_flows,
            run_model,
            timesteps,
            epoch,
            {"get_flow_values": get_flow_values, "vector_field": vector_field},
        )


def get_split_graphs(cgraph: ComputeGraph, dyn_params: list[str] = None) -> SplitGraphs:
    """
    Split the graph into constant, timestep, and init parts.
    Args:
        cgraph: The compute graph to split
        dyn_params: The dynamic parameters to split the graph on

    Returns:
        SplitGraphs: A named tuple containing the constant, timestep, and init compute graphs
    """

    # We are only calibrating these parameters; therefore everything else can be treated as constant
    # within this runner
    if dyn_params is None:
        dyn_params = cgraph.query("parameters.")

    # Get all model based variables (time, state); these are time varying by definition
    time_varying_keys = cgraph.query("model_variables")

    # Include timestep based model variables; time, state
    all_dyn_keys = dyn_params + time_varying_keys

    dyn_cg, constant_cg = cgraph.freeze(all_dyn_keys)

    # Split the graph into timestep and non-constant but time-static init parts
    timestep_cg, init_cg = dyn_cg.freeze(time_varying_keys)

    return SplitGraphs(constant_cg, init_cg, timestep_cg)

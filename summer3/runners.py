from typing import Optional
from .proto import *
from .managed import ManagedArray, ManagedIndex
from .categories import *
from jax import lax, jit, grad, make_jaxpr
import jax
from .utils import Epoch
import pandas as pd
import diffrax as dfx

from summer3.computegraph import ComputeGraph
from summer3.computegraph.types import GraphObject


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


class CompartmentalModelRunner:
    def __init__(
        self,
        model: "CompartmentalModel",
        graph,
        actual_flows,
        run_func,
        timesteps,
        epoch: Optional[Epoch] = None,
    ):
        self.model = model
        self.graph = graph
        self.actual_flows = actual_flows
        self._run_func = run_func
        self._graph_func = self.graph.get_callable(output_all=True)
        self.timesteps = timesteps

        if epoch is None:
            self._time_idx = pd.Index(np.arange(0, timesteps))
        else:
            self._time_idx = epoch.index_to_dti(np.arange(0, timesteps))

    def run(self, init_state, params):
        gathered_res = self._run_func(init_state, params)
        flow_outputs = gathered_res["flows"]
        compartment_outputs = gathered_res["compartments"]
        computed_values = gathered_res["computed_values"]

        cmap = self.model.cmap

        cdata = ManagedArray(
            compartment_outputs,
            dims=["time", "compartment"],
            indices={
                "time": ManagedIndex("time", self._time_idx),
                "compartment": ManagedIndex("compartment", cmap),
            },
        )

        flow_data = {
            flow_key: ManagedArray(
                flow_data,
                dims=["time", "compartment"],
                indices={
                    "time": ManagedIndex("time", self._time_idx),
                    "source": ManagedIndex(
                        "compartment", self.actual_flows[flow_key].src_cmap
                    ),
                    "dest": ManagedIndex(
                        "compartment", self.actual_flows[flow_key].dest_cmap
                    ),
                },
                labellers={"compartment": flow_labeller},
            )
            for flow_key, flow_data in flow_outputs.items()
        }

        # Run a single timestep to get the realised coords of all items in the graph
        # Hopefully some of this disappears in optimization?
        ref_comp_outs = compartment_outputs[0, :]
        cvals = self.model.cmap.zeros().as_managed_array()
        model_variables = {"time": 0, "compartment_values": cvals}
        dyn_vals = self._graph_func(model_variables=model_variables, parameters=params)

        out_cv = {}

        for k, v in computed_values.items():
            ref_ma = dyn_vals[k]
            indices = {"time": ManagedIndex("time", self._time_idx)}
            out_cv[k] = ManagedArray(
                v, dims=["time"] + ref_ma.dims, indices=indices | ref_ma.indices
            )

        return {"compartments": cdata, "flows": flow_data, "computed_values": out_cv}


class CompartmentalModel:
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
            for iadj, adj in enumerate(flow.adjustments):
                if isinstance(adj, GraphObject):
                    adj_key = f"_flow_param_{k}_adj[{iadj}]"
                    adj_param_keys[iadj] = adj_key
                    graph_dict[adj_key] = adj
            actual_flows[k] = flow.actualize(self.cmap, flow_param_key, adj_param_keys)

        return ComputeGraph(graph_dict), actual_flows

    def get_runner(
        self, timesteps, epoch=None, jit=False, computed_values=None
    ) -> CompartmentalModelRunner:
        cgraph, actual_flows = self.actualize_flows()
        cgraphfunc = cgraph.get_callable(output_all=True)
        computed_values = computed_values or []

        def run_model(init_state, params):
            def state_update(comp_vals, i):
                hdata = self.cmap.wrap_data(comp_vals)
                hdata = hdata.as_managed_array()
                model_variables = {"time": i, "compartment_values": hdata}
                dyn_values = cgraphfunc(
                    model_variables=model_variables, parameters=params
                )

                comp_delta = jnp.zeros_like(comp_vals)
                stored_flows = {}
                for k, flow in actual_flows.items():
                    flow_vals = flow.get_flow_vals(hdata, dyn_values)
                    stored_flows[k] = flow_vals
                    if hasattr(flow, "src_cmap"):
                        comp_delta = comp_delta.at[
                            flow.src_cmap.parent_indices
                        ].subtract(flow_vals)
                    if hasattr(flow, "dest_cmap"):
                        comp_delta = comp_delta.at[flow.dest_cmap.parent_indices].add(
                            flow_vals
                        )
                tstep_data = jnp.clip(hdata.data + comp_delta, 0.0)

                stored_dyn = {}
                for k in computed_values:
                    if isinstance(dyn_values[k], ManagedArray):
                        stored_dyn[k] = dyn_values[k].data
                    else:
                        stored_dyn[k] = dyn_values[k]

                return tstep_data, {
                    "compartments": tstep_data,
                    "flows": stored_flows,
                    "computed_values": stored_dyn,
                }

            final, gathered = lax.scan(
                state_update, init_state, xs=jnp.arange(timesteps)
            )
            return gathered

        if jit:
            run_model = jit(run_model, static_argnames=["timesteps"])

        return CompartmentalModelRunner(
            self, cgraph, actual_flows, run_model, timesteps, epoch
        )


class CompartmentalModelODERunner:
    def __init__(
        self,
        model: "CompartmentalModelODE",
        graph,
        actual_flows,
        run_func,
        timesteps,
        epoch: Optional[Epoch] = None,
        funcs=None,
    ):
        self.model = model

        self.graph = graph
        self.actual_flows = actual_flows
        self._run_func = run_func
        self._graph_func = self.graph.get_callable(output_all=True)
        self.timesteps = timesteps
        self.funcs = funcs

        if epoch is None:
            self._time_idx = pd.Index(np.arange(0, timesteps))
        else:
            self._time_idx = epoch.index_to_dti(np.arange(0, timesteps))

    def run(self, init_state, params, solver_args=None):
        solver_args = solver_args or {}
        gathered_res = self._run_func(init_state, params, **solver_args)
        flow_outputs = gathered_res["flows"]
        compartment_outputs = gathered_res["compartments"]
        computed_values = gathered_res["computed_values"]

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
        cvals = self.model.cmap.zeros().as_managed_array()
        model_variables = {"time": 0, "compartment_values": cvals}
        dyn_vals = self._graph_func(model_variables=model_variables, parameters=params)

        out_cv = {}

        for k, v in computed_values.items():
            ref_ma = dyn_vals[k]
            indices = {"time": ManagedIndex("time", self._time_idx)}
            out_cv[k] = ManagedArray(
                v, dims=["time"] + ref_ma.dims, indices=indices | ref_ma.indices
            )

        return {
            "compartments": cdata,
            "flows": flow_data,
            "computed_values": out_cv,
            "aux": gathered_res.get("aux"),
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
            for iadj, adj in enumerate(flow.adjustments):
                if isinstance(adj, GraphObject):
                    adj_key = f"_flow_param_{k}_adj[{iadj}]"
                    adj_param_keys[iadj] = adj_key
                    graph_dict[adj_key] = adj
            actual_flows[k] = flow.actualize(self.cmap, flow_param_key, adj_param_keys)

        return ComputeGraph(graph_dict), actual_flows

    def get_runner(
        self, timesteps, epoch=None, jit=False, computed_values=None
    ) -> CompartmentalModelODERunner:
        cgraph, actual_flows = self.actualize_flows()
        cgraphfunc = cgraph.get_callable(output_all=True)
        computed_values = computed_values or []

        def get_flow_values(t, y, params):
            comp_vals = y
            hdata = self.cmap.wrap_data(comp_vals)
            hdata = hdata.as_managed_array()
            model_variables = {"time": t, "compartment_values": hdata}
            dyn_values = cgraphfunc(model_variables=model_variables, parameters=params)

            stored_flows = {}
            for k, flow in actual_flows.items():
                flow_vals = flow.get_flow_vals(hdata, dyn_values)
                stored_flows[k] = flow_vals

            return stored_flows

        def vector_field(t, y, params):
            comp_vals = y
            hdata = self.cmap.wrap_data(comp_vals)
            hdata = hdata.as_managed_array()
            model_variables = {"time": t, "compartment_values": hdata}
            dyn_values = cgraphfunc(model_variables=model_variables, parameters=params)

            comp_delta = jnp.zeros_like(comp_vals)
            for k, flow in actual_flows.items():
                flow_vals = flow.get_flow_vals(hdata, dyn_values)
                if hasattr(flow, "src_cmap"):
                    comp_delta = comp_delta.at[flow.src_cmap.parent_indices].subtract(
                        flow_vals
                    )
                if hasattr(flow, "dest_cmap"):
                    comp_delta = comp_delta.at[flow.dest_cmap.parent_indices].add(
                        flow_vals
                    )
            return comp_delta

        def run_model(init_state, params, dtmax=1.0):
            term = dfx.ODETerm(vector_field)
            solver = dfx.Dopri5()  # cust
            saveat = dfx.SaveAt(ts=jnp.arange(timesteps))
            stepsize_controller = dfx.PIDController(
                rtol=1e-5, atol=1e-5, dtmax=dtmax
            )  # , dtmax=1.0)

            adjoint = dfx.RecursiveCheckpointAdjoint()
            # adjoint = diffrax.ForwardMode()
            # adjoint = diffrax.DirectAdjoint()
            sol = dfx.diffeqsolve(
                term,
                solver,
                t0=0,
                t1=timesteps,
                throw=False,
                max_steps=int(2 * timesteps),
                dt0=0.1,
                y0=init_state,
                args=params,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                adjoint=adjoint,
            )

            flow_values = jax.vmap(get_flow_values, in_axes=(0, 0, None))(
                sol.ts, sol.ys, params
            )

            return {
                "compartments": sol.ys,
                "flows": flow_values,
                "computed_values": {},
                "aux": sol,
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
            cgraph,
            actual_flows,
            run_model,
            timesteps,
            epoch,
            {"get_flow_values": get_flow_values, "vector_field": vector_field},
        )


class CompartmentalModelODEPL:
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
            for iadj, adj in enumerate(flow.adjustments):
                if isinstance(adj, GraphObject):
                    adj_key = f"_flow_param_{k}_adj[{iadj}]"
                    adj_param_keys[iadj] = adj_key
                    graph_dict[adj_key] = adj
            actual_flows[k] = flow.actualize(self.cmap, flow_param_key, adj_param_keys)

        return ComputeGraph(graph_dict), actual_flows

    def get_runner(
        self, timesteps, epoch=None, jit=False, computed_values=None
    ) -> CompartmentalModelODERunner:
        cgraph, actual_flows = self.actualize_flows()
        cgraphfunc = cgraph.get_callable(output_all=True)
        computed_values = computed_values or []

        def get_flow_values(t, y, params):
            comp_vals = y
            hdata = self.cmap.wrap_data(comp_vals)
            hdata = hdata.as_managed_array()
            model_variables = {"time": t, "compartment_values": hdata}
            dyn_values = cgraphfunc(model_variables=model_variables, parameters=params)

            stored_flows = {}
            for k, flow in actual_flows.items():
                flow_vals = flow.get_flow_vals(hdata, dyn_values)
                stored_flows[k] = flow_vals

            return stored_flows

        def vector_field(t, y, params):
            comp_vals = y
            hdata = self.cmap.wrap_data(comp_vals)
            hdata = hdata.as_managed_array()
            model_variables = {"time": t, "compartment_values": hdata}
            dyn_values = cgraphfunc(model_variables=model_variables, parameters=params)

            comp_delta = jnp.zeros_like(comp_vals)
            for k, flow in actual_flows.items():
                flow_vals = flow.get_flow_vals(hdata, dyn_values)
                if hasattr(flow, "src_cmap"):
                    comp_delta = comp_delta.at[flow.src_cmap.parent_indices].subtract(
                        flow_vals
                    )
                if hasattr(flow, "dest_cmap"):
                    comp_delta = comp_delta.at[flow.dest_cmap.parent_indices].add(
                        flow_vals
                    )
            return comp_delta

        def run_model(init_state, params, dtmax=1.0):
            term = dfx.ODETerm(vector_field)
            solver = dfx.Dopri5()  # cust
            saveat = dfx.SaveAt(ts=jnp.arange(timesteps))
            stepsize_controller = dfx.PIDController(
                rtol=1e-5, atol=1e-5, dtmax=dtmax
            )  # , dtmax=1.0)

            adjoint = dfx.RecursiveCheckpointAdjoint()
            # adjoint = diffrax.ForwardMode()
            # adjoint = diffrax.DirectAdjoint()
            sol = dfx.diffeqsolve(
                term,
                solver,
                t0=0,
                t1=timesteps,
                throw=False,
                max_steps=int(2 * timesteps),
                dt0=0.1,
                y0=init_state,
                args=params,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                adjoint=adjoint,
            )

            flow_values = jax.vmap(get_flow_values, in_axes=(0, 0, None))(
                sol.ts, sol.ys, params
            )

            return {
                "compartments": sol.ys,
                "flows": flow_values,
                "computed_values": {},
                "aux": sol,
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
            cgraph,
            actual_flows,
            run_model,
            timesteps,
            epoch,
            {"get_flow_values": get_flow_values, "vector_field": vector_field},
        )

from proto import *
from jax import lax, jit, grad, make_jaxpr

from summer3.computegraph import ComputeGraph
from summer3.computegraph.types import GraphObject


class NaiveModel:
    def __init__(self, cmap, flows, dyn_params):
        self.cmap = cmap
        self.flows = flows
        self.dyn_params = dyn_params
        self.actual_flows = {}

    def actualize_flows(self):
        for k, v in self.flows.items():
            self.actual_flows[k] = v.actualize(self.cmap)

    def get_runner(self, jit=False):

        self.actualize_flows()

        def run_model(init_state, params, timesteps):
            def state_update(comp_vals, i):
                params["t"] = i
                hdata = self.cmap.wrap_data(comp_vals)
                for k, v in self.dyn_params.items():
                    params[k] = v(hdata, params)

                comp_delta = jnp.zeros_like(comp_vals)
                for k, flow in self.actual_flows.items():
                    flow_vals = flow.get_flow_vals(hdata, params)
                    if hasattr(flow, "src_cmap"):
                        comp_delta = comp_delta.at[flow.src_cmap.indices].subtract(
                            flow_vals
                        )
                    if hasattr(flow, "dest_cmap"):
                        comp_delta = comp_delta.at[flow.dest_cmap.indices].add(
                            flow_vals
                        )
                tstep_data = jnp.clip(hdata.data + comp_delta, 0.0)

                return tstep_data, tstep_data

            final, gathered = lax.scan(
                state_update, init_state, xs=jnp.arange(timesteps)
            )
            return gathered

        if jit:
            run_model = jit(run_model, static_argnames=["timesteps"])

        return run_model


class GraphModelRunner:
    def __init__(self, model, graph, run_func):
        self.model = model
        self.graph = graph
        self._run_func = run_func

    def run(self, init_state, params, timesteps):
        return self._run_func(init_state, params, timesteps)


class GraphModel:
    def __init__(self, cmap, flows: dict[str, TransitionFlow], dyn_params):
        self.cmap = cmap
        self.flows = flows
        self.dyn_params = dyn_params
        self.actual_flows = {}

    def actualize_flows(self):
        graph_dict = {}
        for k, flow in self.flows.items():
            if isinstance(flow.param, GraphObject):
                flow_param_key = f"_flow_param_{k}"
                graph_dict[flow_param_key] = flow.param
            else:
                flow_param_key = None
            self.actual_flows[k] = flow.actualize(self.cmap, flow_param_key)

        return ComputeGraph(graph_dict)

    def get_runner(self, jit=False):
        cgraph = self.actualize_flows()
        cgraphfunc = cgraph.get_callable()

        def run_model(init_state, params, timesteps):
            def state_update(comp_vals, i):
                model_variables = {"time": i, "compartment_values": comp_vals}
                dyn_values = cgraphfunc(
                    model_variables=model_variables, parameters=params
                )
                hdata = self.cmap.wrap_data(comp_vals)

                comp_delta = jnp.zeros_like(comp_vals)
                stored_flows = {}
                for k, flow in self.actual_flows.items():
                    flow_vals = flow.get_flow_vals(hdata, dyn_values)
                    stored_flows[k] = flow_vals
                    if hasattr(flow, "src_cmap"):
                        comp_delta = comp_delta.at[flow.src_cmap.indices].subtract(
                            flow_vals
                        )
                    if hasattr(flow, "dest_cmap"):
                        comp_delta = comp_delta.at[flow.dest_cmap.indices].add(
                            flow_vals
                        )
                tstep_data = jnp.clip(hdata.data + comp_delta, 0.0)

                return tstep_data, {"compartments": tstep_data, "flows": stored_flows}

            final, gathered = lax.scan(
                state_update, init_state, xs=jnp.arange(timesteps)
            )
            return gathered

        if jit:
            run_model = jit(run_model, static_argnames=["timesteps"])

        return GraphModelRunner(self, cgraph, run_model)

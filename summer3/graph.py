from summer3 import computegraph as cg


class Parameter(cg.Variable):
    def __init__(self, key, default):
        super().__init__(key, "parameters")
        self.default = default


class ModelVariable(cg.Variable):
    def __init__(self, key):
        super().__init__(key, "model_variables")
        self.key = key


Time = ModelVariable("time")
CompartmentValues = ModelVariable("compartment_values")


def defer(func, name=None):
    def _proxy(*args, **kwargs):
        return cg.Function(func, args, kwargs, name)

    return _proxy


def label(graph_obj, name):
    graph_obj.node_name = name
    return graph_obj

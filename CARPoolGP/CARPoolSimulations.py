"""
This should be simple class that holds on to the parameters and quantities that come from a simulaiont
"""
import numpy as np

class Simulation:
    def __init__(self, parameters = 0, quantities = 0, lbs=None, ubs=None, fids=None):
        self.parameters = parameters
        self.quantities = quantities
        self.lbs=lbs
        self.ubs=ubs
        self.fids=fids

    @property
    def parameters(self):
        return self._parameters
    @parameters.setter
    def parameters(self, value):
        value = np.atleast_1d(value)
        self.parameter_dimensions = np.shape(value.reshape(len(value), -1))[1]
        self._parameters = value.reshape(-1, self.parameter_dimensions)
    @property
    def quantities(self):
        return self._quantities
    @quantities.setter
    def quantities(self, value):
        self._quantities = value

    def update(self, parameters, quantities):
        parameters = np.atleast_1d(parameters).reshape(-1, self.parameter_dimensions)
        self.parameters = np.concatenate((self.parameters, parameters))
        self.quantities = np.concatenate((self.quantities, quantities))

    def pop(self, index=-1, return_popped=False):
        if return_popped:
            to_return = (self.parameters[index:], self.quantities[index:])
        self.parameters = self.parameters[:index]
        self.quantities = self.quantities[:index]
        if return_popped:
            return to_return

    def normalize_X(self, val, param=None):
        if any(value is None for value in [self.fids, self.lbs, self.ubs]):
            raise ValueError("must define lbs, ubs, fids")
        if param is None:
            return ((val - self.lbs) / (self.ubs-self.lbs))
        else:
            return ((val - self.lbs[param]) / (self.ubs[param] - self.lbs[param]))


    def unnormalize_X(self, val, param=None):
        if any(value is None for value in [self.fids, self.lbs, self.ubs]):
            raise ValueError("must define lbs, ubs, fids")
        if param is None:
            return (val)* (self.ubs-self.lbs) + self.lbs
        else:
            return (val) * (self.ubs[param] - self.lbs[param]) + self.lbs[param]

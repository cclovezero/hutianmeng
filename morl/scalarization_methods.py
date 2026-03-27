import numpy as np
from abc import abstractmethod
import torch

class ScalarizationFunction():
    def __init__(self, num_objs, weights = None, prac_weight = None):
        self.num_objs = num_objs
        if weights is not None:
            self.weights = torch.Tensor(weights)
        else:
            self.weights = None

        if prac_weight is not None:
            self.prac_weight = torch.Tensor(prac_weight)
        else:
            self.prac_weight = None
    
    def update_weights(self, weights, prac_weight = None):
        if weights is not None:
            self.weights = torch.Tensor(weights)
        if prac_weight is not None:
            self.prac_weight = torch.Tensor(prac_weight)

    @abstractmethod
    def evaluate(self, objs):
        pass

class WeightedSumScalarization(ScalarizationFunction):
    def __init__(self, num_objs, weights = None):
        super(WeightedSumScalarization, self).__init__(num_objs, weights)
    
    def update_z(self, z):
        pass

    def evaluate(self, objs):
        return (objs * self.weights).sum(axis = -1)
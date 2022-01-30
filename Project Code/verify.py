from gurobipy import *
from gurobipy import GRB
import numpy as np


class VerificationHelper(object):
    def __init__(self, gmodel):
        self.gmodel = gmodel

    def _dense_vars(self, layer):
        output_size = layer.shape[1]
        epsilon_vars = []
        output_vars = []
        delta_vars = []
        for i in range(output_size):
            epsilon_vars.append(self.gmodel.addVar())
            output_vars.append(self.gmodel.addVar(lb=-GRB.INFINITY))
            delta_vars.append(self.gmodel.addVar(vtype=GRB.BINARY))
        return epsilon_vars, output_vars, delta_vars

    def _dense_vars_ffnn(self, layer):
        output_size = layer.output_shape[1]
        epsilon_vars = []
        output_vars = []
        delta_vars = []
        for i in range(output_size):
            epsilon_vars.append(self.gmodel.addVar())
            output_vars.append(self.gmodel.addVar(lb=-GRB.INFINITY))
            delta_vars.append(self.gmodel.addVar(vtype=GRB.BINARY))
        return epsilon_vars, output_vars, delta_vars

    def _dense_constraints(self, layer, epsilons, inputs, outputs):
        output_size = layer.shape[1]
        weights = layer.T
        
        dotted_outputs = []
        for i in range(output_size):
            #print(weights[i].shape)
            dotted_outputs.append((np.array(weights[i]).dot(inputs)))
            #dotted_outputs.append(np.asarray(np.dot(weights[i],inputs))
            
            
        for i in range(output_size):
            self.gmodel.addConstr(dotted_outputs[i] - outputs[i] <= epsilons[i])
            self.gmodel.addConstr(dotted_outputs[i] - outputs[i] >= -epsilons[i])
            
    def _dense_constraints_ffnn(self, layer, epsilons, inputs, outputs):
        output_size = layer.output_shape[1]
        weights = layer.get_weights()[0].T
        #bias = layer.get_weights()[1]
        dotted_outputs = [weights[i].dot(inputs) for i in range(output_size)]
        for i in range(output_size):
            self.gmodel.addConstr(dotted_outputs[i] - outputs[i] <= epsilons[i])
            self.gmodel.addConstr(dotted_outputs[i] - outputs[i] >= -epsilons[i])

    def _relu_vars(self, relu_dimension):
        relu_vars = []
        for i in range(relu_dimension):
            relu_vars.append(self.gmodel.addVar())
        return relu_vars

    def _relu_vars_ffnn(self, layer):
        output_size = layer.output_shape[1]
        relu_vars = []
        for i in range(output_size):
            relu_vars.append(self.gmodel.addVar())
        return relu_vars

    def _relu_constraints(self, relu_dimension, pre, post, delta):
        for i in range(relu_dimension):
            # self.gmodel.addGenConstrMax(post[i], [pre[i]], 0)
            self.gmodel.addConstr(post[i] >= pre[i])
            self.gmodel.addConstr(post[i] <= pre[i] + 1000 * delta[i])
            self.gmodel.addConstr(post[i] <= 1000 * (1 - delta[i]))

    def _relu_constraints_ffnn(self, layer, pre, post, delta):
        output_size = layer.output_shape[1]
        for i in range(output_size):
            # self.gmodel.addGenConstrMax(post[i], [pre[i]], 0)
            self.gmodel.addConstr(post[i] >= pre[i])
            self.gmodel.addConstr(post[i] <= pre[i] + 1000 * delta[i])
            self.gmodel.addConstr(post[i] <= 1000 * (1 - delta[i]))

    def add_vars(self, layers):
        dense, relu = [], []
        for i in range(len(layers)):
            dense.append(self._dense_vars(layers[i]))
            relu.append(self._relu_vars(layers[i].shape[1]))
        return dense, relu

    def add_vars_ffnn(self, layers):
        dense, relu = [], []
        for i in range(0, len(layers) - 1, 2):
            dense.append(self._dense_vars_ffnn(layers[i]))
            relu.append(self._relu_vars_ffnn(layers[i + 1]))
        dense.append(self._dense_vars_ffnn(layers[-1]))
        return dense, relu

    def add_constraints(self, layers, il, dense, relu):
        for i in range(0, len(dense)):
            e, o, d = dense[i]
            r = relu[i]
            self._dense_constraints(layers[i], e, il, o)
            self._relu_constraints(layers[i].shape[1], o, r, d)
            il = r
        return o

    def add_constraints_ffnn(self, layers, il, dense, relu):
        for i in range(0, len(relu)):
            e, o, d = dense[i]
            r = relu[i]
            self._dense_constraints_ffnn(layers[2 * i], e, il, o)
            self._relu_constraints_ffnn(layers[2 * i + 1], o, r, d)
            il = r
        (e, o, _) = dense[-1]
        self._dense_constraints_ffnn(layers[-1], e, il, o)
        return o

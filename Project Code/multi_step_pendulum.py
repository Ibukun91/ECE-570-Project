#!/usr/bin/env python

import math
import numpy as np
import sys
import argparse

# from keras.utils import plot_model
from pendulum import Pendulum
from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
from keras.models import load_model
from keras_lstm_checker import construct_rnn_from_keras_rnn

from timeit import default_timer as timer
from verify import VerificationHelper
from models import build_model
from lstm_abstractor import LSTMAbstractor
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", default=1, type=int, help="The number of steps to use")
parser.add_argument("-u", "--unroll", default="demand", choices=["start", "demand"], type=str, help="Unrolling method to use.")
parser.add_argument("-a", "--angle", default=70, type=int, help="Can theta > -pi/<angle> in <steps> number of steps?")
ARGS = parser.parse_args()

# Setup some constants.
DEBUG = True

dt = .05  # Seconds between state updates

pendulum = Pendulum()

NUM_STEPS = ARGS.steps
ANGLE = ARGS.angle
np.random.seed(1337)

# Load the pre-trained Keras neural network.
agent_model = build_model()
agent_model.load_weights("/content/drive/My Drive/ECE570/code/weights_pendulum_rnn.h5")   #weights_pendulum_rnn #pendulum_new_weights_rnn-01-0.0000.hdf5
keras_rnn = construct_rnn_from_keras_rnn(agent_model, built_myself=True)

# Also load the sin and cos env models.
sin_model = load_model("/content/drive/My Drive/ECE570/code/sin.h5")
cos_model = load_model("/content/drive/My Drive/ECE570/code/cos.h5")

# Create the LP model and a wrapper to add constraints.
gmodel = Model("Test")
wrapper_gmodel = VerificationHelper(gmodel)
gmodel.Params.LogToConsole = 0
# gmodel.Params.DualReductions = 0
gmodel.Params.MIPGap = 1e-6
gmodel.Params.FeasibilityTol = 1e-7
gmodel.Params.IntFeasTol = 1e-6

# Add the network inputs and the constraints on them.

initialTheta = gmodel.addVar()
initialThetaDot = gmodel.addVar()
network_input = []

sinThetas = []
cosThetas = []
actions = []
start_constrs = timer()
for i in range(NUM_STEPS):

    # Add the env variables.
    dense_sin, relu_sin = wrapper_gmodel.add_vars_ffnn(sin_model.layers)
    dense_cos, relu_cos = wrapper_gmodel.add_vars_ffnn(cos_model.layers)
    gmodel.update()

    if ARGS.unroll == 'start':
        abstractor = LSTMAbstractor(keras_rnn, i + 1, abstraction_type=INPUT_ON_START_ONE_OUTPUT) #keras_rnn
    else:
        abstractor = LSTMAbstractor(keras_rnn, i + 1, abstraction_type=INPUT_ON_DEMAND_ONE_OUTPUT)
    agent_model_ffnn = abstractor.build_abstraction()

    # Add the agent variables to the network.
    dense, relu = wrapper_gmodel.add_vars(agent_model_ffnn.get_layers())

    if i > 0:
        [sin_theta] = wrapper_gmodel.add_constraints_ffnn(sin_model.layers, [theta], dense_sin, relu_sin)
        [cos_theta] = wrapper_gmodel.add_constraints_ffnn(cos_model.layers, [theta], dense_cos, relu_cos)
    else:
        [sin_theta] = wrapper_gmodel.add_constraints_ffnn(sin_model.layers, [initialTheta], dense_sin, relu_sin)
        [cos_theta] = wrapper_gmodel.add_constraints_ffnn(cos_model.layers, [initialTheta], dense_cos, relu_cos)

    sinThetas.append(sin_theta)
    cosThetas.append(cos_theta)

    gmodel.update()

    # Add the constraints for the network itself.
    if i > 0:
        qvals = wrapper_gmodel.add_constraints(agent_model_ffnn.get_layers(), network_input, dense, relu)
    else:
        qvals = wrapper_gmodel.add_constraints(agent_model_ffnn.get_layers(), [cos_theta, sin_theta, initialThetaDot], dense, relu)
    # Set constraints to determine the action taken by the agent.
    o = gmodel.addVar(vtype=GRB.BINARY)
    action = gmodel.addVar(lb=-1,ub=1,vtype=GRB.INTEGER)

    # Compute the action of the agent based on the Q-value with highest value.
    # We currently assume that if Q-values are the same, we end up with MOVE_RIGHT as the action.
    gmodel.addConstr((o == 0) >> (qvals[1] <= qvals[0]))
    gmodel.addConstr((o == 1) >> (qvals[0] <= qvals[1]))

    gmodel.addConstr((o == 0) >> (action == -1))
    gmodel.addConstr((o == 1) >> (action == 1))

	
    

#old syntax for gurobi 7.5.2. Doesn't work on Gurobi 9.5.0
    #if (o == 0):
     #   action = -1
    #elif (o == 1):
     #   action = 1

    actions.append(action)

    # Add constraints for all the epsilons.
    epsilons = quicksum((quicksum(e) for (e, _, _) in dense))
    epsilons += quicksum((quicksum(e) for (e, _, _) in dense_sin))
    epsilons += quicksum((quicksum(e) for (e, _, _) in dense_cos))
    gmodel.addConstr(epsilons <= 1e-7)

    # Create the output functions.
    if i > 0:
        theta_dot = pendulum.theta_dot(theta_dot, sin_theta, action)
        theta = pendulum.theta(theta, theta_dot)
    else:
        theta_dot = pendulum.theta_dot(initialThetaDot, sin_theta, action)
        theta = pendulum.theta(initialTheta, theta_dot)

    if i == 0:
        network_input = [cos_theta, sin_theta, initialThetaDot]
    network_input.extend([cos_theta, sin_theta, theta_dot])

# Constrain the initial angle to small region around pi / 2.
initialTheta.lb = 0.0
initialTheta.ub = math.pi / 64.0

# Constrain the initial speed to small values.
initialThetaDot.lb = 0.0
initialThetaDot.ub = 0.3

# Check if any state can be reached where the angle is outside an arbitrary bound
if NUM_STEPS == 0:
    gmodel.addConstr(initialTheta <= -math.pi / ANGLE)
else:
    gmodel.addConstr(theta <= -math.pi / ANGLE)

# Update the model.
gmodel.update()
end_constrs = timer()
print('Finished adding constraints. Took {}s.'.format(end_constrs - start_constrs))

# plot_model(agent_model, to_file='agent_model.png', show_shapes=True)
# plot_model(sin_model, to_file='sin_model.png', show_shapes=True)
# plot_model(cos_model, to_file='cos_model.png', show_shapes=True)

start = timer()
gmodel.optimize()
end = timer()
if gmodel.status == GRB.OPTIMAL:
    currentThetaDot = initialThetaDot.x
    currentTheta = initialTheta.x
    print("There is a feasible solution found.")
    for i in range(NUM_STEPS):
        print("State ({}, {})".format(currentTheta, currentThetaDot))
        print("Action {}".format(actions[i]))
        currentThetaDot = pendulum.theta_dot(currentThetaDot, sinThetas[i].x, actions[i])
        currentTheta = pendulum.theta(currentTheta, currentThetaDot)
    print("State ({}, {})".format(currentTheta, currentThetaDot))
elif gmodel.status == GRB.CUTOFF or gmodel.status == GRB.INFEASIBLE:
    print("Infeasible: No solution found.")
print("Angle: pi / {}".format(ANGLE))
print("k: {}".format(ANGLE))
print("Time taken (verification): {}s".format(end - start))
print("Number of variables: {}".format(gmodel.NumVars))
print("Number of constraints: {}".format(gmodel.NumConstrs))

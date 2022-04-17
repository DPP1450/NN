from random import random
import math
def initialize_network(shape):
    network = list()
    n = len(shape)
    for i in range (1,n):
        layer = [{'weight': [random() for j in range(shape[i-1] + 1)]} for j in range(shape[i])]
        network.append(layer)
    return network

def activate(weight, inputs):
    activation = weight[-1]
    for i in range(len(weight)-1):
        activation += weight[i]*inputs[i]
    return activation

def transfer(activation):
    return 1.0/(1.0+ math.exp(-activation))

def transfer_derivative(output):
    return output*(1 - output)

def forward_propagate(network,row): #把資料通過每一層計算出output
    inputs = row
    for layer in network:
        new_inputs=[]
        for neuron in layer:
            activation = activate(neuron['weight'], inputs)
            neuron['output']= transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))): #從最後一層往回算
        layer = network[i]
        errors= []
        if i !=len(network)-1: #hidden layer
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weight'][j]*neuron['delta'])
                errors.append(error)
        else: #output layer
            for j in range(len(layer)):
                neuron= layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron=layer[j]
            neuron['delta']= errors[j]* transfer_derivative(neuron['output'])

def update_weight(network,row,l_rate):
    for i in range(len(network)):
        inputs= row[:-1]
        if i!=0:
            inputs=[neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weight'][j] += l_rate *neuron['delta']* inputs[j]
            neuron['weight'][-1] += l_rate* neuron['delta'] #bias
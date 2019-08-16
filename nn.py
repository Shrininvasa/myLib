#!/usr/bin/env python
# coding: utf-8


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def dsigmoid(x):
    return x * (1 - x)
def tanh(x):
    return math.tanh(x)
def dtanh(x):
    return 1 - (x * x)

class ActivationFunction:
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc
sigmoid_af = ActivationFunction(sigmoid, dsigmoid)
tanh_af = ActivationFunction(tanh, dtanh)

import Matrix as m
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.weights_ih = m.Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = m.Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()
        
        self.bias_h = m.Matrix(self.hidden_nodes, 1)
        self.bias_o = m.Matrix(self.output_nodes, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()
        self.learning_rate = 0.1
        self.af = sigmoid_af
        
    def predict(self, input_array):
        inputs = m.Matrix.fromArray(input_array)
        
        hidden = m.Matrix.multiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.mapThis(self.af.func)
        
        output = m.Matrix.multiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.mapThis(self.af.func)
        
        return output.toArray()
    
    def setActivationFunction(self, func = sigmoid_af):
        self.af = func
    
    def train(self, inp_array, target_array):        
        inputs = m.Matrix.fromArray(inp_array)
        
        hidden = m.Matrix.multiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.mapThis(self.af.func)
        
        outputs = m.Matrix.multiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.mapThis(self.af.func)
        targets = m.Matrix.fromArray(target_array)
      
        output_errors = m.Matrix.subtract(targets, outputs)

        gradients = m.Matrix.mapThisStatic(outputs, self.af.dfunc)

        gradients.scale(output_errors)
        gradients.scale(self.learning_rate)
        #self.weights_ih.printMatrix()
        
        hidden_T = m.Matrix.transpose(hidden)
        weights_ho_deltas = m.Matrix.multiply(gradients, hidden_T)
        
        
        self.weights_ho.add(weights_ho_deltas)
        self.bias_o.add(gradients)
        
        who_t = m.Matrix.transpose(self.weights_ho)
        hidden_errors = m.Matrix.multiply(who_t, output_errors)

        hidden_gradient = m.Matrix.mapThisStatic(hidden, self.af.dfunc)

        hidden_gradient.scale(hidden_errors)
        hidden_gradient.scale(self.learning_rate)
        #hidden_gradient.printMatrix()
        
        inputs_T = m.Matrix.transpose(inputs)
        weight_ih_deltas = m.Matrix.multiply(hidden_gradient, inputs_T)
        
        self.weights_ih.add(weight_ih_deltas)
        #weight_ih_deltas.printMatrix()
        self.bias_h.add(hidden_gradient)



if __name__ == "__main__":
    import random as r
    print("Solving the XOR problem using tanh activation function")
    nn = NeuralNetwork(2, 4, 1)
    nn.setActivationFunction(tanh_af)

    input_arr = [[1, 0], [0, 1], [1, 1], [0, 0]]
    target_arr = [[1], [1], [0], [0]]

    print(nn.predict(input_arr[0]))
    print(nn.predict(input_arr[1]))
    print(nn.predict(input_arr[2]))
    print(nn.predict(input_arr[3]))
    print('-' * 80)

    for x in range(25000):
        r_int = int(r.random() * len(input_arr))
        nn.train(input_arr[r_int], target_arr[r_int])


    print('-' * 80)
    print(nn.predict(input_arr[3]))
    print(nn.predict(input_arr[2]))
    print(nn.predict(input_arr[1]))
    print(nn.predict(input_arr[0]))


input()





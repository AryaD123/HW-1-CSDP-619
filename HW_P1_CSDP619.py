"""
Name: Arya Das
Course: CSDP 619
Assignment: HW 1
Date Updated: 09/21/2024
"""
import math
import numpy as np

# Part 1: Implement activation fucntions
e = math.e
print("e: ", e)

def sigmoid(x):
    return 1 / (1 + e**(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Tanh Function and Tanh Derivative
def tanh(x):
    return (e**x - e**(-x)) / (e**x + e**(-x))

def tanh_derivative(x):
    return 1 - tanh(x)**2


# ReLU Function and ReLU Derivative
def relu(x):
    return max(0, x)

def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0


# Leaky ReLU Function and Leaky ReLU Derivative
def leaky_relu(x):
    a = .01
    return max(a*x, x)

def leaky_relu_derivative(x):
    a = .01
    if x > 0:
        return 1
    else:
        return a


#Softmax Function and Softmax Derivative
def softmax(values):
    exp_values = np.exp(values)
    exp_values_sum = np.sum(exp_values)
    return exp_values/exp_values_sum

def softmax_gradient(values):
    return 
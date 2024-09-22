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

# Sigmoid Function and Sigmoid Derivative
x = 0 # Update as needed, or take user input
def sigmoid(x):
    return 1 / (1 + e**(-x))
##sigmoid = 1 / (1 + e**(-x))
#print("Sigmoid: ", sigmoid)
#sigmoid_derivative = sigmoid * (1 - sigmoid)
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
#print("Sigmoid Derivative: ", sigmoid_derivative)


# Tanh Function and Tanh Derivative
#tanh = (e**x - e**(-x)) / (e**x + e**(-x))
#print("Tanh: ", tanh)
def tanh(x):
    return (e**x - e**(-x)) / (e**x + e**(-x))
#tanh_derivative = 1 - tanh**2
#print("Tanh Derivative: ", tanh_derivative)
def tanh_derivative(x):
    return 1 - tanh(x)**2


# ReLU Function and ReLU Derivative
# relu = max(0, x)
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
# e ^ (x - max(x)) / sum(e^(x - max(x))
# = e ^ x / (e ^ max(x) * sum(e ^ x / e ^ max(x)))
# = e ^ x / sum(e ^ x)

def softmax_gradient(values):
    return 
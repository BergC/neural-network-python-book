import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU Activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input using NumPy
        self.output = np.maximum(0, inputs)

# Softmax Activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabitlies
        # We subtract the largest input value to help combat exploding values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabitilies = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabitilies

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 ouput values
dense1 = Layer_Dense(2, 3)

# Create ReLU Activation (to be used with Dense Layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function.
# Takes in output from previous layer.
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Output
print(activation2.output[:5])

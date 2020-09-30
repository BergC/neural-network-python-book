import numpy as np

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)

#
# # Output of current layer
# layer_outputs = []
#
# # For each neuron
# for neuron_weights, neuron_bias in zip(weights, biases):
#     # Zeroed output of given neuron
#     neuron_output = 0
#
#     # For each input and weight to the neuron
#     for n_input, weight in zip(inputs, neuron_weights):
#         # Multiply this input by associated weight
#         # and add to the neuron's output variable
#         neuron_output += n_input * weight
#
#     # Add bias
#     neuron_output += neuron_bias
#
#     # Put neuron's result to the layer's output list
#     layer_outputs.append(neuron_output)
#
# print(layer_outputs)

# import numpy as np
#
# inputs = [1.0, 2.0, 3.0, 2.5]
#
# weights = [[0.2, 0.8, -0.5, 1.0],
#             [0.5, -0.91, 0.26, -0.5],
#             [-0.26, -0.27, 0.17, 0.87]]
#
# biases = [2.0, 3.0, 0.5]
#
# # Because np.dot will return a vector of equal length to our biases, we can simply use
# # the + sign to add values at each index and the result will be a same-length vector.
# layer_outputs = np.dot(weights, inputs) + biases
#
# print(layer_outputs)
#
# a = [1, 2, 3]
# b = [2, 3, 4]
#
# a = np.array([a])
# b = np.array([b]).T
#
# print(np.dot(a, b))
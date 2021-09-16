# This is a sample Python script.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
num_neurons_layer = 10
training_sample_size = 6000
data = pd.read_csv("C:/Users/danie/Downloads/mnist_test.csv.zip")
data = np.array(data)
data_dev = data[0:training_sample_size].T
print(data_dev.size)
y_true = data_dev[0]
# data organized as 10000 * 785
training_data = data_dev[1:785]

training_data = training_data.T / 255
print(training_data.shape)
# doing this would now mean that training_data is now 1000 rows and 784 columns
samples, num_pixels = np.shape(training_data)
step_size = 0.1


# making the matrices for the bias and the weights
# the first hidden layer is organized so that there are 785 rows and then 10 columns, for ten weights
def layer(num_neurons, num_inputs):
    weights = 10 ** -2 * np.random.randn(num_inputs, num_neurons)
    biases = 10 ** -2 * np.random.randn(1, num_neurons)
    return weights, biases


def activation_ReLU(inputs):
    return np.maximum(0, inputs)


def forward_prop(weights1, weights2, biases1, biases2, inputs):
    layer1 = np.dot(inputs, weights1) + biases1
    # layer1 should be a 10 column and 1000 row matrix; 10 columns because there were 10 columns in the weights
    layer1 = activation_ReLU(layer1)
    layer2 = np.dot(layer1, weights2) + biases2
    outputs = softmax_function(layer2)
    return layer1, layer2, outputs


def softmax_function(outputs):
    return np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)


# this is the loss function that will be used
def categorical_cross_entropy(outputs, y_true):
    true_predictions = np.array(outputs[range(len(outputs)), y_true])
    total_loss = np.clip(true_predictions, 10 ** -10, 1 - 10 ** -10)
    total_loss = -np.log(total_loss)
    total_loss = np.mean(total_loss)
    return total_loss


# def d_ReLU(input):
#     return input > 0

def gradient_descent(weights2, outputs, layer1):
    # remember that you don't want to take the dot product because the dot produce involves adding

    outputs[range(samples), y_true] -= 1
    der_loss_WRT_score = outputs

    dweights2 = np.dot(layer1.T, der_loss_WRT_score) / samples
    # WHy organize the dot product this way for dweights2 and not the other way: this is because if you did the other
    # way you would get a matrix that is organized incorrectly;
    # Pay Attention to how the weights matrix is organized: inputs (rows) by predictions/classifications (columns)
    # Dot product this way means that the matrix is organized: neuron by predictions with respect to 1 neuron
    # Dot product the other way means that the matrix is organized: predictions by neuron with respect to 1 prediction
    # the first would have to be chosen because remember that the organization must match the weights to be added
    dbiases2 = np.sum(der_loss_WRT_score, axis=0, keepdims=True) / samples
    # for the dbiases2, you have to make sure that you add up all of the derivatives WRT the score among the samples
    # you are not adding up the derivatives among one sample, that wouldn't make sense because you are finding overall

    # remember the true_predictions has 1000 rows and is a vertical array
    # this would mean that der_loss_WRT_score also is the same shape
    # therefore you can perform the dot product because the row is 1 element long, multiplied by the
    # der_loss_WRT_L1 = d_ReLU(np.dot(der_loss_WRT_score, weights2.T))
    # this doesn't allow you to keep the original value of element if its greater than 0 which you want
    der_loss_WRT_L1 = np.dot(der_loss_WRT_score, weights2.T)

    der_loss_WRT_L1[layer1 <= 0] = 0
    # THIS LINE SHORTCUTS WHAT YOU WOULD NORMALLY DO
    # You would have to take the derivative of the activation function for the layer and then multiply that by
    # Layer 1 is activated!

    # print(der_loss_WRT_L1)
    # der_loss_WRT_L1 = d_ReLU(der_loss_WRT_L1)
    dweights1 = np.dot(training_data.T, der_loss_WRT_L1) / samples
    dbiases1 = np.sum(der_loss_WRT_L1, axis=0, keepdims=True) / samples
    return dweights1, dweights2, dbiases1, dbiases2


def make_predictions(num_samples, weights1, weights2, biases1, biases2):
    segmented_data = data[2001: 2001 + num_samples].T
    labels = segmented_data[0]
    raw_data = segmented_data[1: 785].T
    layer1_values = activation_ReLU(np.dot(raw_data, weights1) + biases1)
    layer2_values = np.dot(layer1_values, weights2) + biases2
    for i in range(num_samples):
        current_image = data[training_sample_size + 1 + i]
        current_image = current_image[1:785]
        print("Actual:", labels[i], "Predicted:", np.argmax(layer2_values, axis=1)[i])

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()


weights1, biases1 = layer(10, num_pixels)
weights2, biases2 = layer(10, 10)
# print(weights1.shape, weights2.shape, biases1.shape, biases2.shape)

for i in range(2000):
    layer1, layer2, outputs = forward_prop(weights1, weights2, biases1, biases2, training_data)

    total_loss = categorical_cross_entropy(outputs, y_true)
    if i % 100 == 0:
        print("iteration:", i, "loss: ", total_loss)
        print("Accuracy:", np.mean(np.argmax(outputs, axis=1) == y_true))

    # if i == 499:
    #     for index in range(1000):
    #         print(np.argmax(outputs, axis = 1)[index], y_true[index])
    dweights1, dweights2, dbiases1, dbiases2 = gradient_descent(weights2, outputs, layer1)
    # print(weights1)
    weights1 -= step_size * dweights1
    # print(weights1)
    weights2 -= step_size * dweights2
    biases1 -= step_size * dbiases1
    biases2 -= step_size * dbiases2

make_predictions(training_sample_size, weights1, weights2, biases1, biases2)
# layer1, layer2, outputs = forward_prop(weights1, weights2, biases1, b
# iases2, training_data)
# total_loss, true_predictions = categorical_cross_entropy(outputs, y_true)
#
#
# dweights1, dweights2, dbiases1, dbiases2 = gradient_descent(weights2, outputs, layer1)
# weights1 -= step_size * dweights1
# weights2 -= step_size * dweights2
# biases1 -= step_size * dbiases1
# biases2 -= step_size * dbiases2


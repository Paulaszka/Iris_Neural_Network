import pickle
from functions import *
import random
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, layer_sizes, want_bias):
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        self.use_bias = want_bias
        if want_bias:
            self.biases = []
            for y in self.sizes[1:]:
                self.biases.append(np.random.uniform(-1, 1, (y, 1)))
        else:
            self.biases = []
            for y in self.sizes[1:]:
                self.biases.append(np.zeros((y, 1)))

        self.weights = []
        for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            self.weights.append(np.random.uniform(-1, 1, (y, x)))

        self.delta_wi = []
        for w in self.weights:
            self.delta_wi.append(np.zeros(w.shape))

    def train(self, training_data, epochs, early_stopping_error, learning_rate, momentum, shuffle, hops):
        error_log = ""
        train = list(training_data)

        for epoch in range(epochs):
            if shuffle == 1:
                random.shuffle(train)

            self.update(train, learning_rate, momentum)
            if early_stopping_error != -1 and early_stopping_error >= self.epoch_error(train):
                print("Osiagnieto pozadany poziom bledu.")
                with open('data/train_logs.csv', 'w') as file:
                    file.write(error_log)
                return

            if epoch % hops == 0:
                print(f"Epoka: {epoch}")
                error_log += f"{epoch}, {self.epoch_error(train)}\n"
        with open('data/train_logs.csv', 'w') as file:
            file.write(error_log)

    def update(self, single_batch, learning_rate, momentum):
        bias_gradient = []
        for b in self.biases:
            bias_gradient.append(np.zeros(b.shape))

        weight_gradient = []
        for w in self.weights:
            weight_gradient.append(np.zeros(w.shape))

        for x, y in single_batch:
            bias_gradient_delta, weight_gradient_delta = self.back_propagation(x, y)

            for i in range(len(bias_gradient)):
                bias_gradient[i] += bias_gradient_delta[i]

            for i in range(len(weight_gradient)):
                weight_gradient[i] += weight_gradient_delta[i]

        new_delta_wi = []
        flag = True
        counter = 0
        for i, (wi, gw) in enumerate(zip(self.delta_wi, weight_gradient)):
            if flag:
                updated_wi = momentum * 0 - (learning_rate / len(single_batch)) * gw
                counter += 1
                if counter == 2:
                    flag = False
            elif not flag:
                updated_wi = momentum * (self.weights[i] - self.weights[i-1]) - (learning_rate / len(single_batch)) * gw
            new_delta_wi.append(updated_wi)
        self.delta_wi = new_delta_wi

        updated_weights = []
        for w, wi in zip(self.weights, self.delta_wi):
            new_weights = w + wi
            updated_weights.append(new_weights)
        self.weights = updated_weights

        if self.use_bias:
            updated_biases = []
            for b, gb in zip(self.biases, bias_gradient):
                new_biases = momentum * b - (learning_rate / len(single_batch)) * gb
                updated_biases.append(new_biases)
            self.biases = updated_biases

    def forward_propagation(self, x_data):
        for bias, weight in zip(self.biases, self.weights):
            x_data = sigmoid(np.dot(weight, x_data) + bias)
        return x_data

    def back_propagation(self, x, y):
        bias_gradient = []
        for b in self.biases:
            bias_gradient.append(np.zeros(b.shape))

        weight_gradient = []
        for w in self.weights:
            weight_gradient.append(np.zeros(w.shape))

        activation = x
        activations = [x]
        weighted_layer = []
        for b, w in zip(self.biases, self.weights):
            weighted_neuron = np.dot(w, activation) + b
            weighted_layer.append(weighted_neuron)
            activation = sigmoid(weighted_neuron)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(activations[-1])
        bias_gradient[-1] = delta
        weight_gradient[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            activation = activations[-layer]
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_derivative(activation)
            bias_gradient[-layer] = delta
            weight_gradient[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return bias_gradient, weight_gradient

    def epoch_error(self, train_data):
        error = 0
        for x, y in train_data:
            error += self.calculate_error(self.forward_propagation(x), y)
        return error / len(train_data)

    @staticmethod
    def cost_derivative(output_activations, expected_output):
        return output_activations - expected_output

    @staticmethod
    def calculate_error(expected, output):
        return np.mean(np.power(expected - output, 2))

    def save_network(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_network(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    def plot_training_error(self):
        with open('data/train_logs.csv', 'r') as file:
            data = file.readlines()
        epochs = []
        errors = []
        for line in data:
            epoch, error = map(float, line.strip().split(','))
            epochs.append(epoch)
            errors.append(error)
        plt.plot(epochs, errors, marker='', linestyle='-', color='#fa39b6')
        plt.xlabel('Epoka')
        plt.ylabel('Błąd')
        plt.grid(True)
        plt.show()

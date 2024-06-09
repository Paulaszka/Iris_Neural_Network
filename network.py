import pickle
from functions import *
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    f = sigmoid(x)
    return f * (1 - f)


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

        self.velocity = []
        for w in self.weights:
            self.velocity.append(np.zeros(w.shape))

    def plot_training_error(self):
        with open('trainError.csv', 'r') as file:
            data = file.readlines()
        epochs = []
        errors = []
        for line in data:
            epoch, error = map(float, line.strip().split(','))
            epochs.append(epoch)
            errors.append(error)
        plt.plot(epochs, errors, marker='', linestyle='-')
        plt.title('Błąd popełniony w kolejnych epokach nauki sieci')
        plt.xlabel('Epoka')
        plt.ylabel('Błąd')
        plt.grid(True)
        plt.show()

    def feedforward(self, x_data):
        for bias, weight in zip(self.biases, self.weights):
            x_data = sigmoid(np.dot(weight, x_data) + bias)
        return x_data

    def train(self, training_data, epochs, precision, batch_size, learning_rate, momentum, shuffle, hops,
              validation_data, debug=False):
        error_log = ""
        train = list(training_data)
        valid = list(validation_data)

        for epoch in range(epochs):
            if shuffle:
                random.shuffle(train)

            batch_list = []
            for k in range(0, len(train), batch_size):
                single_batch = train[k:k + batch_size]  # wycinamy od k do k+batch_size
                batch_list.append(single_batch)

            for single_batch in batch_list:
                self.update(single_batch, learning_rate, momentum)

            if epoch % hops == 0:
                if valid:
                    test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                                    for (x, y) in valid]
                    num_correct = sum(int(x == y) for (x, y) in test_results)
                    current_precision = num_correct / len(valid)
                    epoch_error = self.epoch_error(valid)
                    if debug:
                        print(epoch_error)
                        print(f"Epoch {epoch} : {num_correct} / {len(valid)} Precision: {current_precision}")
                    if current_precision >= precision:
                        print("Desired precision reached, stopping training.")
                        with open('trainError.csv', 'w') as file:
                            file.write(error_log)
                        return
                else:
                    if debug:
                        print(f"Epoch {epoch} complete")
                error_log += f"{epoch}, {self.epoch_error(train)}\n"
        with open('trainError.csv', 'w') as file:
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

        new_velocity = []
        for v, gw in zip(self.velocity, weight_gradient):
            updated_v = momentum * v - (learning_rate / len(single_batch)) * gw
            new_velocity.append(updated_v)
        self.velocity = new_velocity

        updated_weights = []
        for w, v in zip(self.weights, self.velocity):
            new_weights = w + v
            updated_weights.append(new_weights)
        self.weights = updated_weights

        if self.use_bias:
            updated_biases = []
            for b, gb in zip(self.biases, bias_gradient):
                new_biases = momentum * b - (learning_rate / len(single_batch)) * gb
                updated_biases.append(new_biases)
            self.biases = updated_biases

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
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(weighted_layer[-1])
        bias_gradient[-1] = delta
        weight_gradient[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            weighted_neuron = weighted_layer[-layer]
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_derivative(weighted_neuron)
            bias_gradient[-layer] = delta
            weight_gradient[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return bias_gradient, weight_gradient

    def epoch_error(self, train_data):
        error = 0
        for x, y in train_data:
            error += self.calculate_error(self.feedforward(x), y)
        return error / len(train_data)

    @staticmethod
    def cost_derivative(output_activations, expected_output):
        return output_activations - expected_output

    def save_network(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_network(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def calculate_error(expected, output):
        return np.mean(np.power(expected - output, 2))
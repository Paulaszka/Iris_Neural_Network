import pickle
from functions import *
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    f = sigmoid(x)
    return f * (1 - f)


class Network(object):
    def __init__(self, layer_sizes, useBias):
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        self.useBias = useBias
        if useBias:
            self.biases = []
            for y in layer_sizes[1:]:
                self.biases.append(np.random.uniform(-1, 1, (y, 1)))
        else:
            self.biases = []
            for y in layer_sizes[1:]:
                self.biases.append(np.zeros((y, 1)))

        self.weights = []
        for x, y in zip(layer_sizes[:-1], layer_sizes[1:]):
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

    def train(self, training_data, epochs, precision, batch_size, learning_rate, momentum, shuffle, error_epoch,
              validation_data, debug=False):
        error_log = ""
        training_data = list(training_data)
        validation_data = list(validation_data)

        for epoch in range(epochs):
            if shuffle:
                random.shuffle(training_data)

            batch_list = []
            for k in range(0, len(training_data), batch_size):
                single_batch = training_data[k:k + batch_size]
                batch_list.append(single_batch)

            for single_batch in batch_list:
                self.update(single_batch, learning_rate, momentum)

            if epoch % error_epoch == 0:
                if validation_data:
                    test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                                    for (x, y) in validation_data]
                    num_correct = sum(int(x == y) for (x, y) in test_results)
                    current_precision = num_correct / len(validation_data)
                    epoch_error = self.epoch_error(validation_data)
                    if debug:
                        print(epoch_error)
                        print(f"Epoch {epoch} : {num_correct} / {len(validation_data)} Precision: {current_precision}")
                    if current_precision >= precision:
                        print("Desired precision reached, stopping training.")
                        with open('trainError.csv', 'w') as file:
                            file.write(error_log)
                        return
                else:
                    if debug:
                        print(f"Epoch {epoch} complete")
                error_log += f"{epoch}, {self.epoch_error(training_data)}\n"
        with open('trainError.csv', 'w') as file:
            file.write(error_log)

    def update(self, single_batch, learning_rate, momentum):
        gradient_b = []
        for b in self.biases:
            gradient_b.append(np.zeros(b.shape))

        gradient_w = []
        for w in self.weights:
            gradient_w.append(np.zeros(w.shape))

        for x, y in single_batch:
            delta_gradient_b, delta_gradient_w = self.backpropagation(x, y)

            for i in range(len(gradient_b)):
                gradient_b[i] += delta_gradient_b[i]

            for i in range(len(gradient_w)):
                gradient_w[i] += delta_gradient_w[i]

        new_velocity = []
        for v, gw in zip(self.velocity, gradient_w):
            updated_v = momentum * v - (learning_rate / len(single_batch)) * gw
            new_velocity.append(updated_v)
        self.velocity = new_velocity

        new_weights = []
        for w, v in zip(self.weights, self.velocity):
            updated_w = w + v
            new_weights.append(updated_w)
        self.weights = new_weights

        if self.useBias:
            new_biases = []
            for b, gb in zip(self.biases, gradient_b):
                updated_b = b - (learning_rate / len(single_batch)) * gb
                new_biases.append(updated_b)
            self.biases = new_biases

    def backpropagation(self, x, y):
        gradient_b = []
        for b in self.biases:
            gradient_b.append(np.zeros(b.shape))

        gradient_w = []
        for w in self.weights:
            gradient_w.append(np.zeros(w.shape))

        activation = x
        activations = [x]
        weighted_layer = []
        for b, w in zip(self.biases, self.weights):
            weighted_neuron = np.dot(w, activation) + b
            weighted_layer.append(weighted_neuron)
            activation = sigmoid(weighted_neuron)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(weighted_layer[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            weighted_neuron = weighted_layer[-layer]
            sigmoid_prime = sigmoid_derivative(weighted_neuron)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_prime  # blad dla aktualnej warstwy, +1 następna warstwa
            gradient_b[-layer] = delta
            gradient_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())  # -1 poprzednia warstwa
        return gradient_b, gradient_w

    def epoch_error(self, train_data):
        error = 0
        for x, y in train_data:
            error += self.calculate_error(self.feedforward(x), y)
        return error / len(train_data)

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations - y  # wynik na ostatniej warstwie - oczekiwany wynik

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def calculate_error(expected, output):
        return np.mean(np.power(expected - output, 2))

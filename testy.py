import network
import unittest
import pandas as pd
from functions import *
from numpy import *


class MyTestCase(unittest.TestCase):
    def test_training(self):
        data_set = int(input("Podaj cyfre\n"
                             "1 - irysy\n"
                             "2 - autoenkoder\n"))

        def prepare_data_auto(data):
            data = np.array(data)
            target_values = []
            for genre in data:
                if genre[-1] == 0:
                    target_values.append([1, 0, 0, 0])
                elif genre[-1] == 1:
                    target_values.append([0, 1, 0, 0])
                elif genre[-1] == 2:
                    target_values.append([0, 0, 1, 0])
                elif genre[-1] == 3:
                    target_values.append([0, 0, 0, 1])
            x_array = data[:, :-1]
            target_values = np.array(target_values)
            combined_data = []
            for x, y in zip(x_array, target_values):
                combined_data.append((x.reshape(-1, 1), y.reshape(-1, 1)))
            return combined_data

        if data_set == 1:
            data_list = pd.read_csv('data/data.csv', header=None)
            data_list = data_list.sample(frac=1).reset_index(drop=True)
            train_raw = data_list.tail(120)
            test_raw = data_list.head(30)
            test_raw.to_csv('data/test.csv', index=False, header=False)
            train_data = prepare_data_bin(train_raw)
            test_data = prepare_data_bin(test_raw)

        if data_set == 2:
            train_data = [(array([[1], [0], [0], [0]]), array([[1], [0], [0], [0]])),
                          (array([[0], [1], [0], [0]]), array([[0], [1], [0], [0]])),
                          (array([[0], [0], [1], [0]]), array([[0], [0], [1], [0]])),
                          (array([[0], [0], [0], [1]]), array([[0], [0], [0], [1]]))]

            test_data = [(array([[1], [0], [0], [0]]), array([[1], [0], [0], [0]])),
                         (array([[0], [1], [0], [0]]), array([[0], [1], [0], [0]])),
                         (array([[0], [0], [1], [0]]), array([[0], [0], [1], [0]])),
                         (array([[0], [0], [0], [1]]), array([[0], [0], [0], [1]]))]

        neuron_number_list = [4, 2, 4]
        bias = 1
        early_stopping_epoch = 1000
        early_stopping_error = 0.05
        learning_rate = 0.2
        momentum = 0.9
        want_random = 1
        hops = 10

        net = network.Network(neuron_number_list, want_bias=(False if bias == 0 else True))

        net.train(train_data, epochs=early_stopping_epoch, early_stopping_error=early_stopping_error,
                  learning_rate=learning_rate, momentum=momentum, shuffle=want_random, hops=hops)

        test_network(net, test_data)

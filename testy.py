import network
import unittest
import pandas as pd
from functions import *


class MyTestCase(unittest.TestCase):
    def test_training(self):
        data_set = int(input("Podaj cyfre\n"
                             "1 - irysy\n"
                             "2 - autoenkoder\n"))

        if data_set == 1:
            data_list = pd.read_csv('data/data.csv', header=None)
            data_list = data_list.sample(frac=1).reset_index(drop=True)
            train_raw = data_list.tail(120)
            test_raw = data_list.head(30)
            test_raw.to_csv('data/test.csv', index=False, header=False)
            train_data = prepare_data(train_raw)
            test_data = prepare_data(test_raw)

        if data_set == 2:
            x_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            y_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            xy_data = []
            for x, y in zip(x_data, y_data):
                xy_data.append((x.reshape(-1, 1), y.reshape(-1, 1)))
            test_data = xy_data
            train_data = xy_data

        neuron_number_list = [4, 3]
        bias = 1
        early_stopping_epoch = 300
        early_stopping_error = -1
        learning_rate = 0.2
        momentum = 0.6
        want_random = 1
        hops = 10

        net = network.Network(neuron_number_list, want_bias=(False if bias == 0 else True))

        net.train(train_data, epochs=early_stopping_epoch, early_stopping_error=early_stopping_error,
                  learning_rate=learning_rate, momentum=momentum, shuffle=want_random, hops=hops)

        test_network(net, test_data)

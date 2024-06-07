import network
import unittest
import warnings
import logging
import pandas as pd
import random

from functions import *

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


class MyTestCase(unittest.TestCase):
    def test_training(self):
        data_set = int(input("Podaj cyfre\n"
                             "1 - irysy\n"
                             "2 - autoenkoder\n"))

        if data_set == 1:
            train_raw = pd.read_csv('data/data.csv', header=None)
            test_raw = pd.read_csv('data/test.csv', header=None)
            train_data = prepare_data(train_raw)
            test_data = prepare_data(test_raw)
            valid = random.choices(train_data, k=int(len(train_data) / 3))
            random.shuffle(valid)

        if data_set == 2:
            x_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            y_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            xy_data = []
            for x, y in zip(x_data, y_data):
                xy_data.append((x.reshape(-1, 1), y.reshape(-1, 1)))
            test_data = xy_data
            train_data = xy_data
            valid = xy_data

        neuron_number_list = [4, 3]
        bias = 1
        early_stopping_epoch = 1000
        early_stopping_error = 0.05
        learning_rate = 0.2
        momentum = 0.6
        want_random = 1
        hops = 10

        net = network.Network(neuron_number_list, useBias=(False if bias == 0 else True))

        net.train(train_data, epochs=early_stopping_epoch, precision=early_stopping_error, batch_size=10,
                  learning_rate=learning_rate, momentum=momentum, shuffle=want_random, error_epoch=hops,
                  validation_data=valid, debug=True)
        confusion(net, test_data)

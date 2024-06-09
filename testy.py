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
            # test_x = [row[:-1] for row in test_data]
            # test_y = [row[-1] for row in test_data]
            valid_data = random.choices(train_data, k=int(len(train_data) / 5))
            random.shuffle(valid_data)


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

        net = network.Network(neuron_number_list, want_bias=(False if bias == 0 else True))

        net.train(train_data, epochs=early_stopping_epoch, early_stopping_error=early_stopping_error,
                  learning_rate=learning_rate, momentum=momentum, shuffle=want_random, hops=hops)
        # confusion(net, test_data)
        # print("Test: ")
        # print(test_data)
        # print("TextX:")
        # print(test_x)
        predicted_labels = []
        true_labels = []

        for index in range(len(test_data)):
            test_row = test_data[index]
            output = net.feedforward(test_row[0])
            expected = test_row[1]
            true_labels.append(np.argmax(expected))
            predicted_labels.append(np.argmax(output))

        # output = net.feedforward(test_x)
        print("true labels", true_labels)
        print("predicted labels", predicted_labels)
        test_logs(net, true_labels, predicted_labels)
        net.plot_training_error()

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
            print(type(test_data))
            test_x = [row[:-1] for row in test_data]
            test_y = [row[-1] for row in test_data]
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

        net = network.Network(neuron_number_list, want_bias=(False if bias == 0 else True))

        net.train(train_data, epochs=early_stopping_epoch, precision=early_stopping_error, batch_size=10,
                  learning_rate=learning_rate, momentum=momentum, shuffle=want_random, hops=hops,
                  validation_data=valid, debug=True)
        # confusion(net, test_data)
        output = net.feedforward(test_x)

        array_data = np.array(output)

        def prepare_type_list(y_pred):
            max_indices = []
            for row in y_pred:
                counter = 0
                maximum = 0
                index_max = 0
                for i in range(len(row)):
                    for value in row[i]:
                        if value > maximum:
                            maximum = value
                            index_max = i
                temp_list = [0] * len(row)
                temp_list[index_max] = 1
                max_indices[counter] = temp_list
                counter += 1
            return max_indices

        lista1 = prepare_type_list(output)

        print(lista1)






        print("result\n",array_data)

        test_logs(net, test_y, output )

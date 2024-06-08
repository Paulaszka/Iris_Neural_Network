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
            prepared_data = pd.read_csv('data/data.csv', header=None)
            # test_raw = pd.read_csv('data/test.csv', header=None)
            #prepared_data = prepare_data(data)
            # Pobierz liczbę wierszy w ramce danych
            num_rows = prepared_data.shape[0]

            # Utwórz listę indeksów wierszy
            row_indices = list(range(num_rows))

            # Przemieszaj indeksy
            random.shuffle(row_indices)

            # Wybierz wiersze w kolejności przemieszanych indeksów
            shuffled_data = prepared_data.iloc[row_indices]

            print(shuffled_data)



            train_data = shuffled_data[:90]
            test_data = shuffled_data[90:90 + 45]
            valid_data = shuffled_data[90 + 45:]

            test_x = test_data.iloc[:, :-1]  # Wybierz wszystkie kolumny oprócz ostatniej jako cechy
            test_y = test_data.iloc[:, -1]

            test_data.to_csv("data/test.csv", index=False, header=False)


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
                  validation_data=valid_data, debug=True)
        # confusion(net, test_data)
        output = net.feedforward(test_x)

        # lista1 = prepare_type_list(output)

        # print(lista1)

        # print("y", test_y)

        test_logs(net, test_y, output)

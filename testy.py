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
            row_indices = list(range(prepared_data.shape[0]))
            random.shuffle(row_indices)
            shuffled_data = prepared_data.iloc[row_indices]

            first_part = prepared_data.iloc[row_indices[:90]]  # Pierwsza część: 90 elementów
            second_part = prepared_data.iloc[row_indices[90:125]]  # Druga część: 35 elementów
            third_part = prepared_data.iloc[row_indices[125:]]  # Trzecia część: 15 elementów

            # Zapisz każdą część do osobnego pliku CSV
            first_part.to_csv('data/train_part.csv', index=False, header=None)
            second_part.to_csv('data/test_part.csv', index=False, header=None)
            third_part.to_csv('data/valid_part.csv', index=False, header=None)

            train_data = pd.read_csv('data/train_part.csv', header=None)
            train_data = prepare_data(train_data)
            test_raw = pd.read_csv('data/test_part.csv', header=None)
            valid_data = pd.read_csv('data/valid_part.csv', header=None)
            valid_data = prepare_data(valid_data)

            print(test_raw)

            # train_raw = pd.read_csv('data/train_part.csv', header=None)
            # test_raw = pd.read_csv('data/test_part.csv', header=None)
            # valid_raw = pd.read_csv('data/valid_part.csv', header=None)
            # print(test_raw)

            # train_data = prepare_data(train_raw)
            # test_data = prepare_data(test_raw)
            # valid_data = prepare_data(valid_raw)
            # print(test_data)

            num_rows = test_raw.shape[0]
            num_cols = test_raw.shape[1]

            x_test = np.zeros((num_rows, num_cols - 1))

            y_test = []

            for i in range(num_rows):
                for j in range(num_cols - 1):
                    x = test_raw.iloc[i, j]  # Access by integer index (more efficient)
                    x_test[i][j] = x
                y = test_raw.iloc[i, -1]  # Access the last column for y
                y_test.append(y)

            print("x_test: ", x_test)
            print("y_test: ", y_test)

            # # Podziel dane na x i y
            # for row in first_part:
            #     x = row.iloc[:-1]  # Get all elements except the last
            #     y = row.iloc[-1]
            #     x_test.append(x)
            #     y_test.append(y)

            #x_test = np.array(x_test)
            #y_test = np.array(y_test)

            #print()



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
        output = net.feedforward(x_test)

        # lista1 = prepare_type_list(output)

        # print(lista1)

        # print("y", test_y)

        test_logs(net, y_test, output)

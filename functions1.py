import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



# - - - FUNKCJE PRZYGOTOWUJACE DANE - - -

def prepare_data(file_path):
    data_file = pd.read_csv(file_path, delimiter=';')
    mapping = {'Graduate': 0, 'Dropout': 1, 'Enrolled': 2}
    data_file = data_file.replace(mapping).infer_objects(copy=False)
    return data_file


def oversample_set(train_set, flag=False):
    x = train_set[train_set.columns[:-1]].values
    y = train_set[train_set.columns[-1]].values

    if flag:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    connected = np.hstack((x, np.reshape(y, (-1, 1))))
    return connected, x, y


# - - - FUNKCJA TRENUJACA - - -

def train_model(neuron_number_list, x_train, y_train_encoded, x_valid, y_valid_encoded, learning_rate,
                batch_size, momentum, want_bias, early_stopping_error, epochs, hops, nn_model):

    c =2+2


# - - - FUNKCJE DO BLEDOW I LOGÓW - - -

def calculate_global_error(y_test, y_pred_bin):
    counter = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred_bin[i]:
            counter += 1

    result = round(counter / len(y_test), 8)
    return result


def calculate_individual_error(y_test, y_pred_bin, types_list):
    individual_error_list = [0] * len(types_list)
    individual_correct_list = [0] * len(types_list)
    num_of_individuals = [0] * len(types_list)
    correct = 0

    for i in range(len(y_test)):
        for j in range(len(types_list)):
            if y_test[i] != y_pred_bin[i] and y_test[i] == types_list[j]:
                individual_error_list[j] += 1
            elif y_test[i] == y_pred_bin[i] and y_test[i] == types_list[j]:
                individual_correct_list[j] += 1

    for i in range(len(y_test)):
        for j in range(len(types_list)):
            if y_test[i] == types_list[j]:
                num_of_individuals[j] += 1

    for i in range(len(individual_correct_list)):
        correct += individual_correct_list[i]

    # print(types_list, "jakie rodzaje")
    # print(num_of_individuals, "ile elementow kazdego rodzaju")
    # print(individual_error_list, "ile bledow kazdego rodzaju")

    for i in range(len(individual_error_list)):
        individual_error_list[i] = individual_error_list[i] / num_of_individuals[i]

    # print(individual_error_list, "procent bledow kazdego rodzaju")

    return individual_error_list, individual_correct_list, correct


def result_logs(y_test, y_pred, indiv_correct, correct, filename):
    with open(filename, 'w') as plik:
        plik.write("POROWNANIE WYNIKOW\nLiczba poprawnie sklasyfikowanych elementow:\n")
        plik.write(str(correct))
        plik.write("\nZ podzialem na klasy:\n")
        for element in indiv_correct:
            plik.write(f"{element}\n")
        plik.write("Wyniki testowe - Wyniki przewidywane\n")
        for test, pred in zip(y_test, y_pred):
            plik.write(f"{test} - {pred}\n")


# - - - FUNKCJE DO WYKRESÓW - - -

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid()
    plt.show()


def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()


# - - - INNE FUNKCJE - - -

def prepare_type_list(y_pred):
    max_indices = []
    for row in y_pred:
        max_index = row.argmax()
        max_indices.append(max_index)
    return np.array(max_indices)


def get_all_weights(model):
    all_weights = []
    for layer in model.layers[::-1]:
        weights = layer.get_weights()
        all_weights.append(weights[0])
    return convert_to_list(all_weights)


def convert_to_list(nested_arrays):
    output_list = []
    for array in nested_arrays:
        row_lists = [list(row) for row in array]

        output_list.append(row_lists)
    return output_list

import numpy as np
import matplotlib.pyplot as plt


# - - - FUNKCJE AKTYWACJI - - -
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# - - - FORMATOWANIE / ZMIANA TYPU DANYCH - - -
def prepare_type_list(y_pred):
    max_indices = []
    for row in y_pred:
        for sub_row in row:
            lista = [0] * 3
            max_index = sub_row.argmax()
            lista[max_index] = 1
            max_indices.append(lista)
        return np.array(max_indices)


def prepare_bin_type_list(y_pred):
    max_indices = []
    for row in y_pred:
        max_index = row.argmax()
        max_indices.append(max_index)
    return np.array(max_indices)


def prepare_data(data):
    data = np.array(data)
    target_values = []
    for genre in data:
        if genre[-1] == 0:
            target_values.append([1, 0, 0])
        elif genre[-1] == 1:
            target_values.append([0, 1, 0])
        elif genre[-1] == 2:
            target_values.append([0, 0, 1])
    x_array = data[:, :-1]
    target_values = np.array(target_values)
    combined_data = []
    for x, y in zip(x_array, target_values):
        combined_data.append((x.reshape(-1, 1), y.reshape(-1, 1)))
    return combined_data


# - - - FUNKCJE LOGUJACE - - -
def test_logs(network, test, result, global_error, sum_error):
    types_list = []
    for i in test:
        if all(not np.array_equal(i, existing) for existing in types_list):
            types_list.append(i)

    indiv_correct, correct, confusion_matrix, precision, recall, f_measure = calculate_to_logs(test, result, types_list)
    global_error = global_error / len(test)
    for i in range(len(sum_error)):
        sum_error[i] = sum_error[i] / len(test)
    with open("data/test_logs.txt", 'w') as plik:
        plik.write("WARTOSCI Z CZESCI TESTOWEJ\n")
        plik.write("POROWNANIE WYNIKOW\n\nLiczba poprawnie sklasyfikowanych elementow:\n")
        plik.write(str(correct))
        plik.write("\n\nZ podzialem na klasy:\n")
        for element in indiv_correct:
            plik.write(f"{element}\n")
        plik.write("\nWyniki testowe - Wyniki przewidywane\n")
        for test, pred in zip(test, result):
            plik.write(f"{test} - {pred}\n")
        plik.write("\nWagi:")
        for wiersz in network.weights:
            for kolumna in wiersz:
                plik.write(f"\n")
                for waga in kolumna:
                    plik.write(f"{waga} ")
        plik.write("\n\nBlad dla calej sieci: " + str(global_error))
        plik.write("\n\nBlad dla poszczegolych wyjsc: ")
        for i in range(len(sum_error)):
            plik.write(f"\nWyjscie {i}: {sum_error[i]}")
        plik.write("\n\nMacierz pomylek: \n")

        for i in range(len(confusion_matrix)):
            plik.write(f"{confusion_matrix[i]} \n")

        for i in range(len(types_list)):
            plik.write(f"\nKlasa {i}: Precision: {precision[i]} Recall: {recall[i]} F-measure: {f_measure[i]}")


def calculate_to_logs(y_test, y_pred, types_list):
    correct = 0
    types_list_bin_bin = prepare_bin_type_list(types_list)
    individual_correct_list = [0] * len(types_list_bin_bin)

    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            correct += 1
            individual_correct_list[y_test[i]] += 1

    confusion_matrix = [[0 for _ in range(len(types_list_bin_bin))] for i in range(len(types_list_bin_bin))]
    for t, r in zip(y_test, y_pred):
        confusion_matrix[t][r] += 1
    precision = [0] * len(types_list_bin_bin)
    recall = [0] * len(types_list_bin_bin)
    f_measure = [0] * len(types_list_bin_bin)

    for i in range(len(types_list_bin_bin)):
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[j][i] for j in range(len(types_list_bin_bin))) - tp
        fn = sum(confusion_matrix[i][j] for j in range(len(types_list_bin_bin))) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f_measure[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[
            i]) > 0 else 0
    return individual_correct_list, correct, confusion_matrix, precision, recall, f_measure


def error_logs(expected, output):
    errors = [0] * len(expected)
    for i in range(len(expected)):
        errors[i] = np.power(expected[i] - output[i], 2)
    return errors

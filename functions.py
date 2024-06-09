import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import network


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


def draw(precision, recall, f_measure):
    fig, ax = plt.subplots()
    names = ["precision", "recall", "f_measure"]
    counts = [precision, recall, f_measure]
    bar_labels = ['red', 'blue', 'orange']
    bar_colors = ['tab:red', 'tab:blue', 'tab:orange']
    ax.bar(names, counts, label=bar_labels, color=bar_colors)
    ax.set_ylabel('Percentage')
    ax.set_title('Results')
    plt.show()


def simulate(layers, train, valid, test):
    bias = False
    epochs = 1000
    error = 1.0
    sim_net = network.Network(layers, want_bias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10, 0.9, 0.0, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)

    sim_net = network.Network(layers, want_bias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10, 0.6, 0.0, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)

    sim_net = network.Network(layers, want_bias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10, 0.2, 0.0, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)

    sim_net = network.Network(layers, want_bias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10, 0.9, 0.6, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)

    sim_net = network.Network(layers, want_bias=(False if bias == 0 else True))
    sim_net.train(train, epochs, error, 10, 0.2, 0.9, 1, 10, valid)
    sim_net.plot_training_error()
    confusion(sim_net, test)


def confusion(network1, test):
    predicted_labels = []
    true_labels = []
    logs = "Wagi:\n" + str(network1.weights)
    logs += "\n\nWejscia obciazajece:\n" + str(network1.biases)
    general_error = 0.0
    for index in range(len(test)):
        test_row = test[index]
        output = network1.feedforward(test_row[0])
        expected = test_row[1]
        true_labels.append(np.argmax(expected))
        predicted_labels.append(np.argmax(output))
        error = network1.calculate_error(expected, output)
        general_error += error
        logs += "Wzorzec wejsciowy:\n" + str(test_row[0]) + "\n"
        logs += "Wzorzec wyjsciowy:\n" + str(expected) + "\n"
        logs += "Uzyskane wyjscia:\n" + str(output) + "\n"
        logs += "Wynik klasyfikacji: " + str(np.argmax(output) + 1) + "\n"
        logs += "Blad wyjsciowy: " + str(error) + "\n\n"

    logs += "Calkowity blad wyjsciowy: " + str(general_error) + "\n"

    with open("stats.txt", 'a') as file:
        file.write(logs)

    matrix = confusion_matrix(true_labels, predicted_labels)
    print("\nMacierz pomyłek:")
    print(matrix)
    recall = []
    i = 0
    for x in matrix:
        tmp = 0
        for a in x:
            tmp += a
        recall.append(x[i] / tmp)
        i += 1
    p = [np.array([matrix[x][y] for x in range(len(matrix))]) for y in range(len(matrix))]
    p = np.array([np.sum(x) for x in p])
    precision = []
    for x, y in zip(np.diag(matrix), p):
        if y == 0:
            precision.append(0)
        else:
            precision.append(x / y)
    f_measure = []
    for x, y in zip(precision, recall):
        if y == 0 or x == 0:
            f_measure.append(0)
        else:
            f_measure.append(2 * x * y / (x + y))

    print("\nPrecyzja (Precision):", precision)
    print("Czułość (Recall):", recall)
    print("Miara F (F-measure):", f_measure)


def test_logs(network, test, result):
    types_list = []
    for i in test:
        if all(not np.array_equal(i, existing) for existing in types_list):
            types_list.append(i)

    result_bin = prepare_type_list(result)
    
    test_bin_bin = prepare_bin_type_list(test)
    result_bin_bin = prepare_bin_type_list(result_bin)

    indiv_correct, correct, confusion_matrix, precision, recall, f_measure = calculate_to_logs(test_bin_bin, result_bin_bin, types_list)
    global_error = 0
    error = 0
    with open("data/test_logs.txt", 'w') as plik:
        plik.write("WARTOSCI Z CZESCI TESTOWEJ\n")
        plik.write("POROWNANIE WYNIKOW\nLiczba poprawnie sklasyfikowanych elementow:\n")
        plik.write(str(correct))
        plik.write("\nZ podzialem na klasy:\n")
        for element in indiv_correct:
            plik.write(f"{element}\n")
        plik.write("Wyniki testowe - Wyniki przewidywane\n")
        for test, pred in zip(test_bin_bin, result_bin_bin):
            plik.write(f"{test} - {pred}\n")
        plik.write("\nWagi: " + str(network.weights))
        for i in range(len(test_bin_bin)):
            # error = calculate_error(test, result)
            # global_error += error
            error += 1
        plik.write("\nBlad dla calej sieci: " + str(global_error))
        plik.write("\nMacierz pomylek: \n")

        for i in range(len(confusion_matrix)):
            plik.write(f"{confusion_matrix[i]} ")

        for i in range(len(types_list)):
            plik.write(f"\n Klasa {i}: Precision: {precision[i]} Recall: {recall[i]} F-measure: {f_measure[i]}")


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
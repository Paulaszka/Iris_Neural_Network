from functions import *
from main_handler import *
import numpy as np
import pandas as pd
import warnings
import logging


pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

early_stopping_error = early_stopping_epoch = 0
neuron_number_list = []

# - - - WYBOR TRYBU - - -

print("Wybierz tryb\n"
      "1 - tryb nauki\n"
      "2 - tryb testowania")
mode = one_two_input()

# - - - TRYB NAUKI - - -

if mode == 1:
    print("\nPodaj sposob pobrania danych.\n"
          "1 - Wczytanie sieci z pliku.\n"
          "2 - Podanie parametrow w konsoli.")
    data_mode = one_two_input()
    if data_mode == 1:
        nn_model = keras.saving.load_model("data_files/nn_model.h5")

    elif data_mode == 2:
        nn_model = None
        print("\nOkresl liczbe warstw w sieci neuronowej.")
        layer_number = int_input()
        neuron_number_list = neuron_list_input(layer_number, neuron_number_list)

    print("\nWybierz warunek stopu (czas zakonczenia nauki).\n"
          "1 - ilosc epok\n"
          "2 - poziom bledu")
    stop_type = one_two_input()

    print("\nPodaj wartosc parametru.")
    if stop_type == 1:
        early_stopping_epoch = int_input()
    elif stop_type == 2:
        early_stopping_error = float_input()

    print("\nCzy chcesz uwzględniać wartość wejścia obciążającego (bias)?\n"
          "1 - tak\n"
          "2 - nie")
    want_bias = one_two_input()

    print("\nPodaj wartosc wspolczynnika nauki.")
    learning_rate = float_input()

    print("\nPodaj wartosc wspolczynnika momentum.")
    momentum = float_input()

    print("\nCzy chcesz podac wartosc czestotliwosci (skok epok) zapisywania wartosci globalnego bledu do pliku?\n"
          "(Domyslna wartosc - 10)\n"
          "1 - tak\n"
          "2 - nie")
    want_hops = one_two_input()
    if want_hops == 1:
        hops = int_input()
    if want_hops == 2:
        hops = 10

    print("\nCzy chcesz prezentowac wzorce treningowe w losowej kolejnosci?\n"
          "1 - tak\n"
          "2 - nie")
    want_random = one_two_input()

    # - - - NAUKA - - -

    data_list_train = prepare_data("data/data.csv")
    train, test = np.split(data_list_train.sample(frac=1), [int((2/3) * len(data_list_train))])
    test.to_csv("data/test.csv", index=False, header=False)

    if want_random == 1:
        train = train.sample(frac=1)

    train, x_train, y_train = oversample_set(train, True)
    y_train_encoded = to_categorical(y_train, neuron_number_list[-1])




    nn_model.save("data/nn_model.pkl")

# - - - TRYB TESTOWANIA - - -

elif mode == 2:
    data_list_test = pd.read_csv("data/test.csv")

    x_test = data_list_test[data_list_test.columns[1:]].values
    y_test = data_list_test[data_list_test.columns[0]].values

    types_list = []
    for i in y_test:
        if i not in types_list:
            types_list.append(i)

    types_list.sort()

    nn_model = keras.saving.load_model("data_files/nn_model.h5")

    y_pred = nn_model.predict(x_test)
    y_pred_bin = prepare_type_list(y_pred)

    report = classification_report(y_test, y_pred_bin)
    conf_matrix = confusion_matrix(y_test, y_pred_bin)

    global_error = calculate_global_error(y_test, y_pred_bin)
    individual_error_list, individual_correct_list, correct = calculate_individual_error(y_test, y_pred_bin, types_list)
    weights_list = get_all_weights(nn_model)

    file1 = "data_files/testing_logs.log"
    file2 = "data_files/testing_results.log"
    test_logs(global_error, individual_error_list, weights_list, report, conf_matrix, file1)
    result_logs(y_test, y_pred_bin, individual_correct_list, correct, file2)

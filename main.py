import network
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

# - - - WYBOR ZBIORU DANYCH ORAZ TRYBU - - -

print("Wybierz zestaw danych\n"
      "1 - irysy\n"
      "2 - autoenkoder")
data_set = one_two_input()

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
        c = 2
        # nn_model = keras.saving.load_model("data_files/nn_model.h5")

    elif data_mode == 2:
        nn_model = None
        layer_number = int_input("\nOkresl liczbe warstw ukrytych w sieci neuronowej.")
        neuron_number_list = neuron_list_input(layer_number, neuron_number_list)
        neuron_number_list.append()
# TODO tutaj trzeba dodac jedna warstwe na koniec listy i dac przygotowanie danych na gore

    print("\nWybierz warunek stopu (czas zakonczenia nauki).\n"
          "1 - ilosc epok\n"
          "2 - poziom bledu")
    stop_type = one_two_input()

    early_stopping_epoch = 1000
    early_stopping_error = 0  # TODO sprawdzic czy ten blad jest okej

    if stop_type == 1:
        early_stopping_epoch = int_input("\nPodaj liczbe epok: ")
    elif stop_type == 2:
        early_stopping_error = float_input("\nPodaj pozadana poziom bledu: ")

    bias = int_input("\nPodaj wartość wejścia obciążającego (bias): ")

    learning_rate = 1
    momentum = 1
    while not (0 <= learning_rate < 1 and 0 <= momentum < 1):
        learning_rate = float_input("\nPodaj wartosc wspolczynnika nauki: ")
        momentum = float_input("\nPodaj wartosc wspolczynnika momentum: ")

    hops = int_input("\nPodaj wartosc czestotliwosci (skok epok) zapisywania do pliku: ")

    print("\nCzy chcesz prezentowac wzorce treningowe w losowej kolejnosci?\n"
          "1 - tak\n"
          "2 - nie")
    want_random = one_two_input()

    # - - - NAUKA - - -

    if data_set == 1:
        data_list_train = pd.read_csv("data/data.csv")
        train, valid, test = np.split(data_list_train.sample(frac=1), [int(0.6 * len(data_list_train)),
                                                                       int(0.8 * len(data_list_train))])
        test.to_csv("data/test.csv", index=False, header=False)

    if data_set == 2:
        x_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        y_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        xy_data = []
        for x, y in zip(x_data, y_data):
            xy_data.append((x.reshape(-1, 1), y.reshape(-1, 1)))  # zmieniamy je na pionowe

        test = train = valid = xy_data  # dane testowe, treningowe i walidacyjne sa takie same

    net = network.Network(neuron_number_list, useBias=(False if bias == 0 else True))

    net.train(train, epochs=early_stopping_epoch, precision=early_stopping_error, batch_size=10,
              learning_rate=learning_rate, momentum=momentum, shuffle=want_random, error_epoch=hops,
              validation_data=valid, debug=True)


# - - - TRYB TESTOWANIA - - -

elif mode == 2:
    data_list_test = pd.read_csv("data/test.csv")



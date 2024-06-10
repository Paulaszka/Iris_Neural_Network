from main_handler import *
from functions import *
import pandas as pd
import network


# - - - WYBOR ZBIORU DANYCH ORAZ TRYBU - - -

print("Wybierz zestaw danych\n"
      "1 - irysy\n"
      "2 - autoenkoder")
data_set = one_two_input()

if data_set == 1:
    data_list = pd.read_csv('data/data.csv', header=None)
    data_list = data_list.sample(frac=1).reset_index(drop=True)
    train_raw = data_list.tail(120)
    test_raw = data_list.head(30)
    test_raw.to_csv('data/test.csv', index=False, header=False)
    train_data = prepare_data(train_raw)
    test_data = prepare_data(test_raw)

if data_set == 2:
    x_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    xy_data = []
    for x, y in zip(x_data, y_data):
        xy_data.append((x.reshape(-1, 1), y.reshape(-1, 1)))
    test_data = xy_data
    train_data = xy_data

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
        net = network.Network.load_network("data/network.pkl")  # TODO sprawdzic czy wczytywanie dziala

    elif data_mode == 2:
        layer_number = int_input("\nOkresl liczbe warstw ukrytych w sieci neuronowej.")
        neuron_number_list = [len(train_data[0][0])]
        for i in range(layer_number):
            neuron_number_list.append(int(input("Podaj liczbe neuronow w " + str(i + 1) + " warstwie ukrytej: ")))
        neuron_number_list.append(len(train_data[0][1]))

    print("\nWybierz warunek stopu (czas zakonczenia nauki).\n"
          "1 - ilosc epok\n"
          "2 - poziom bledu")
    stop_type = one_two_input()

    early_stopping_epoch = 1000
    early_stopping_error = -1.0

    if stop_type == 1:
        early_stopping_epoch = int_input("\nPodaj liczbe epok: ")
    elif stop_type == 2:
        early_stopping_error = float_input("\nPodaj pozadany poziom bledu: ")

    bias = int_input("\nPodaj wartość wejścia obciążającego (bias): ")

    learning_rate = -1
    momentum = -1
    while not (0 <= learning_rate <= 1 and 0 <= momentum <= 1):
        learning_rate = float_input("\nPodaj wartosc wspolczynnika nauki: ")
        momentum = float_input("\nPodaj wartosc wspolczynnika momentum: ")

    hops = int_input("\nPodaj wartosc czestotliwosci (skok epok) zapisywania do pliku: ")

    print("\nCzy chcesz prezentowac wzorce treningowe w losowej kolejnosci?\n"
          "1 - tak\n"
          "2 - nie")
    want_random = one_two_input()

    # - - - NAUKA - - -

    net = network.Network(neuron_number_list, want_bias=(False if bias == 0 else True))

    net.train(train_data, epochs=early_stopping_epoch, early_stopping_error=early_stopping_error,
              learning_rate=learning_rate, momentum=momentum, shuffle=want_random, hops=hops)
    net.save_network("data/network.pkl")


# - - - TRYB TESTOWANIA - - -

elif mode == 2:
    net2 = network.Network.load_network("data/network.pkl")
    test_network(net2, test_data)

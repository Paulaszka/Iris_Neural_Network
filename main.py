from main_handler import *
from functions import *
import random
import pandas as pd


# - - - WYBOR ZBIORU DANYCH ORAZ TRYBU - - -

print("Wybierz zestaw danych\n"
      "1 - irysy\n"
      "2 - autoenkoder")
data_set = one_two_input()

if data_set == 1:
    train_raw = pd.read_csv('data/data.csv', header=None)
    test_raw = pd.read_csv('data/test.csv', header=None)
    train_data = prepare_data(train_raw)
    test_data = prepare_data(test_raw)
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
    early_stopping_error = -1.0  # TODO sprawdzic czy ten blad jest okej

    if stop_type == 1:
        early_stopping_epoch = int_input("\nPodaj liczbe epok: ")
    elif stop_type == 2:
        early_stopping_error = float_input("\nPodaj pozadany poziom bledu: ")

    bias = int_input("\nPodaj wartość wejścia obciążającego (bias): ")

    learning_rate = -1
    momentum = -1
    while learning_rate < 0 and momentum < 0:
        learning_rate = float_input("\nPodaj wartosc wspolczynnika nauki: ")
        momentum = float_input("\nPodaj wartosc wspolczynnika momentum: ")

    hops = int_input("\nPodaj wartosc czestotliwosci (skok epok) zapisywania do pliku: ")

    print("\nCzy chcesz prezentowac wzorce treningowe w losowej kolejnosci?\n"
          "1 - tak\n"
          "2 - nie")
    want_random = one_two_input()

    # - - - NAUKA - - -

    net = network.Network(neuron_number_list, want_bias=(False if bias == 0 else True))

    net.train(train_data, epochs=early_stopping_epoch, precision=early_stopping_error, batch_size=10,
              learning_rate=learning_rate, momentum=momentum, shuffle=want_random, hops=hops,
              validation_data=None, debug=True)


# - - - TRYB TESTOWANIA - - -

elif mode == 2:
    data_list_test = pd.read_csv("data/test.csv")
    net = network.Network.load_network("data/network.pkl")

    combined_test_data = 1  # TODO testowanie
    net.plot_training_error()
    confusion(net, combined_test_data)
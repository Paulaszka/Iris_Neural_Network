from main_handler import *
from functions import *


early_stopping_error = early_stopping_epoch = 0
neuron_number_list = []

# - - - WYBOR ZBIORU DANYCH ORAZ TRYBU - - -

print("Wybierz zestaw danych\n"
      "1 - irysy\n"
      "2 - autoenkoder")
data_set = one_two_input()

if data_set == 1:
    train = pd.read_csv('data/data.csv', header=None)
    test = pd.read_csv('data/test.csv', header=None)
    combined_train_data = prepare_data(train)
    combined_test_data = prepare_data(test)
    valid = random.choices(combined_train_data, k=int(len(combined_train_data) / 3))
    random.shuffle(valid)


if data_set == 2:
    x_array = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_array = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    combined_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(x_array, y_array)]
    combined_test_data = combined_data
    combined_train_data = combined_data
    validation_data = combined_data

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
        layer_number = int_input("\nOkresl liczbe warstw ukrytych w sieci neuronowej.")
        neuron_number_list = [len(combined_train_data[0][0])]
        for i in range(layer_number):
            neuron_number_list.append(int(input("Podaj liczbe neuronow w " + str(i + 1) + " warstwie ukrytej: ")))

        neuron_number_list.append(len(combined_train_data[0][1]))

    print("\nWybierz warunek stopu (czas zakonczenia nauki).\n"
          "1 - ilosc epok\n"
          "2 - poziom bledu")
    stop_type = one_two_input()

    early_stopping_epoch = 1000
    early_stopping_error = 1.1  # TODO sprawdzic czy ten blad jest okej

    if stop_type == 1:
        early_stopping_epoch = int_input("\nPodaj liczbe epok: ")
    elif stop_type == 2:
        early_stopping_error = float_input("\nPodaj pozadany poziom bledu: ")

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

    net = network.Network(neuron_number_list, useBias=(False if bias == 0 else True))

    net.train(combined_train_data, epochs=early_stopping_epoch, precision=early_stopping_error, batch_size=10,
              learning_rate=learning_rate, momentum=momentum, shuffle=want_random, error_epoch=hops,
              validation_data=valid, debug=True)


# - - - TRYB TESTOWANIA - - -

elif mode == 2:
    data_list_test = pd.read_csv("data/test.csv")



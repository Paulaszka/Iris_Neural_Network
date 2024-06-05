def one_two_input():
    correct_value = True
    while correct_value:
        variable = int(input("Podaj 1 lub 2: "))
        correct_value = False
        if variable not in [1, 2]:
            print("Niepoprawny wybor, sprobuj jeszcze raz.")
            correct_value = True
    return variable


def int_input():
    correct_value = True
    while correct_value:
        try:
            variable = int(input("Podaj liczbe calkowita: "))
            correct_value = False
        except ValueError:
            print("Niepoprawny wybor, sprobuj jeszcze raz.")
    return variable


def neuron_list_input(layer_number, neuron_number_list):
    correct_value = True
    for i in range(layer_number):
        print("Podaj liczbe neuronow w warstwie ", i+1)
        while correct_value:
            try:
                neuron_number1 = int(input("Liczba neuronoów: "))
                neuron_number_list.append(neuron_number1)
                correct_value = False
            except ValueError:
                print("Niepoprawny wybor, sprobuj jeszcze raz.")
        correct_value = True
    return neuron_number_list


def float_input():
    correct_value = True
    while correct_value:
        try:
            variable = float(input("Podaj liczbe zmiennoprzecinkowa: "))
            correct_value = False
        except ValueError:
            print("Niepoprawny wybor, sprobuj jeszcze raz.")
    return variable


def one_two_three_input():
    correct_value = True
    while correct_value:
        variable = int(input("Podaj 1, 2 lub 3: "))
        correct_value = False
        if variable not in [1, 2, 3]:
            print("Niepoprawny wybor, sprobuj jeszcze raz.")
            correct_value = True
    return variable

def one_two_input():
    correct_value = True
    while correct_value:
        variable = int(input("Podaj 1 lub 2: "))
        correct_value = False
        if variable not in [1, 2]:
            print("Niepoprawny wybor, sprobuj jeszcze raz.")
            correct_value = True
    return variable


def int_input(message):
    correct_value = True
    while correct_value:
        try:
            variable = int(input(message))
            correct_value = False
        except ValueError:
            print("Sprobuj jeszcze raz. Podaj liczbe calkowita.")
    return variable


def float_input(message):
    correct_value = True
    while correct_value:
        try:
            variable = float(input(message))
            correct_value = False
        except ValueError:
            print("Sprobuj jeszcze raz. Podaj liczbe zmiennoprzecinkowa.")
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

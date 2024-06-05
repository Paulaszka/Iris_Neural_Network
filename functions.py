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

    batch_size = 128

    if nn_model is None:
        bias = True
        if want_bias == 2:
            bias = False

        nn_model = tf.keras.Sequential()

        # wariant 1
        # nn_model.add(tf.keras.layers.Dense(neuron_number_list[0], activation="sigmoid",
        #                                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
        #                                    use_bias=True, input_dim=36))
        #
        # # Dodanie pozostałych warstw
        # for i in range(1, len(neuron_number_list)):
        #     nn_model.add(tf.keras.layers.Dense(neuron_number_list[i], activation="sigmoid",
        #                                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1,
        #                                                                                               maxval=1),
        #                                        use_bias=True))
        # wariant 2 (oryginalny)
        for i in range(len(neuron_number_list)):
            nn_model.add(tf.keras.layers.Dense(neuron_number_list[i], activation="sigmoid",
                                               kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1,
                                                                                                      maxval=1),
                                               use_bias=bias))

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=momentum)
    nn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    csv_logger = tf.keras.callbacks.CSVLogger("data_files/training.log", separator=",", append=False)

    early_stopping = None
    if early_stopping_error != 0:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, verbose=0,
                                                          restore_best_weights=True, min_delta=early_stopping_error)
        history = nn_model.fit(x_train, y_train_encoded, epochs=5000, batch_size=batch_size,
                               validation_data=(x_valid, y_valid_encoded), verbose=0, callbacks=[early_stopping,
                                                                                                 csv_logger])
    elif epochs != 0:
        history = nn_model.fit(x_train, y_train_encoded, epochs=epochs, batch_size=batch_size,
                               validation_data=(x_valid, y_valid_encoded), verbose=0, callbacks=[csv_logger])

    with open("data_files/training.log", 'r') as log_file, open("data_files/training_logs.log", 'w') as data_file:
        epoch_count = 1
        data_file.write(f"BLAD POPELNIONY PRZEZ SIEC CO {hops} EPOK\n\n")
        first_line = True
        for line in log_file:
            values = []
            if first_line:
                first_line = False
            elif epoch_count % 2 == 1:
                values.append(line.split(','))
                loss_value = values[0][2]

                if epoch_count % (hops*2) == 1:
                    data_file.write(f"{loss_value}\n")

            epoch_count += 1

    plot_loss(history)
    plot_accuracy(history)

    return nn_model, history


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


def test_logs(global_error, indiv_error, weights, report, conf_matrix, filename):
    with open(filename, 'w') as plik:
        plik.write("WIELKOSCI Z CZESCI TESTOWEJ\n\nRaport\n")
        plik.write(report)

        plik.write("\nMacierz pomylek\n")
        plik.write(str(conf_matrix))

        plik.write("\n\nBlad dla calego wzorca\n")
        plik.write(str(global_error))

        plik.write("\n\nLista bledow na poszczegolnych wyjsciach \n")
        for element in indiv_error:
            plik.write(f"{element}\n")

        plik.write("\nWagi neuronow wyjsciowych i ukrytych \n")
        for element in weights:
            plik.write(f"{element}\n")


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

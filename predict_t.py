import time
from math import exp
from datetime import datetime
import math
from numpy import mean
import pandas as pd


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def power(my_list):
    return [x**2 for x in my_list]


def perhitungan_t(time_list, magnitudo_list):
    max_value = max(magnitudo_list)
    min_value = min(magnitudo_list)
    max_index = magnitudo_list.index(max_value)
    magnitudo_list_pop = magnitudo_list[:]
    magnitudo_list_pop.pop(max_index)

    if max_index == 0:
        # proses no 1
        tdelta = 0
        # proses no 8
        kenaikan = 0
    else:
        # proses no 1
        s1 = time_list[max_index - 1]
        s2 = time_list[max_index]
        # FMT = '%H:%M:%S'
        # tdate = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
        tdate = s2 - s1
        tdelta = tdate.total_seconds() / 60
        # proses no 8
        slices_list = magnitudo_list[:max_index + 1][:]
        kenaikan = 0
        for k in range(len(slices_list)):
            if k is not len(slices_list) - 1:
                if slices_list[k] < slices_list[k + 1]:
                    kenaikan = kenaikan + 1

    # proses no 2
    mean_all = mean(magnitudo_list)
    mean_pop = mean(magnitudo_list_pop)
    # proses no 3
    Sum = sum(magnitudo_list_pop)
    Pangkat = 4.8 + (1.5 * max_value)
    Pow = math.pow(Sum, Pangkat)
    energy = (math.sqrt(Pow)) / 1000000000
    # proses no 4
    b_value = 0.4342944819 / (mean_all - mean_pop)
    # proses no 5
    kurung = math.pow((max_value - mean_pop), 2)
    totalnya_mean = (1 / 14) * kurung
    mean_square = math.sqrt(totalnya_mean)
    # proses no 6
    max_diff = max_value - min_value
    # proses no 7
    magnitudo_list_pop_pow = power(magnitudo_list_pop)
    kurung_awal = (1 / 13) * sum(magnitudo_list_pop_pow)
    kurung_kedua = math.pow(mean_pop, 2)
    akar_koef = math.sqrt((kurung_awal - kurung_kedua))
    koef = akar_koef / mean_pop

    return [tdelta, mean_pop, energy, b_value, mean_square, max_diff, koef, kenaikan]


def sort_list(tanggal, waktu, magnitudo):
    waktu_strp = list()
    for x in range(len(waktu)):
        try:
            FMT = '%Y-%m-%d %H:%M:%S'
            waktu_strp.append(datetime.strptime(tanggal + ' ' + waktu[x], FMT))
        except:
            FMT = '%Y-%m-%d %H:%M'
            waktu_strp.append(datetime.strptime(tanggal + ' ' + waktu[x], FMT))
    df = pd.DataFrame(list(zip(waktu_strp, magnitudo)),columns=['date', 'mag'])
    df = df.sort_values(by=['date'], ascending=True)
    return list(df["date"]), list(df["mag"])


def main_predict_t(tanggal, waktu, magnitudo, network, minmax):
    print('--------------')
    waktu_asc, magnitudo_asc = sort_list(tanggal, waktu, magnitudo)
    print(waktu_asc)
    print(magnitudo_asc)
    list_prediksi = perhitungan_t(waktu_asc, magnitudo_asc)
    print(list_prediksi)
    minmax = minmax[:-1]
    print(minmax)
    for i in range(len(list_prediksi)):
        list_prediksi[i] = (list_prediksi[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    print(list_prediksi)
    prediction = predict(network, list_prediksi)
    print(prediction)
    # print(network)
    print()
    return prediction

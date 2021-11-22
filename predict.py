import time
from math import exp


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    print("ini activa", activation)
    for i in range(len(weights) - 1):
        print("act", activation)
        activation += weights[i] * inputs[i]
        print("act pros", activation)
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        print("tahapan awal")
        print("lay", layer)
        for neuron in layer:
            print("neu", neuron)
            print(inputs)
            print('-----------')
            activation = activate(neuron['weights'], inputs)
            print("ac for", activation)
            print("sebelum tf",neuron['output'])
            neuron['output'] = transfer(activation)
            print("sesudah tf", neuron['output'])
            new_inputs.append(neuron['output'])
            print("new", new_inputs)
        inputs = new_inputs
        print("inp", inputs)
    return inputs


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    print("out", outputs)
    return outputs.index(max(outputs))


def main_predict(date_time, lintang, bujur, kedalaman, network, minmax):
    print('-----------')
    timestamp = time.mktime(date_time.timetuple())
    list_prediksi = [float(lintang), float(bujur), float(kedalaman), float(timestamp)]
    print(list_prediksi)
    minmax = minmax[:-1]
    print(minmax)
    for i in range(len(list_prediksi)):
        list_prediksi[i] = (list_prediksi[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    print(list_prediksi)
    prediction = predict(network, list_prediksi)
    print("pr", prediction)
    print()
    return prediction

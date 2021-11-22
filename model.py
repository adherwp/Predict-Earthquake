from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# my_formatter = "{0:.2f}"
list_weight = list()
list_weight.clear()
# Load CSV
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Cari min and max values di tiap column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset kolom ke range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split dataset ke k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Evaluate algoritma menggunakan cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    # global my_formatter
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    precisions = list()
    recalls = list()
    f1scores = list()
    output_n_global = 0
    for fold in folds:
        train_set = list(folds)
        # print(train_set)
        train_set.remove(fold)
        # print(train_set)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        print(len(train_set))
        print(len(test_set))
        print(train_set)
        print(test_set)
        predicted, output_n = algorithm(output_n_global, train_set, test_set, *args)
        if output_n_global == 0:
            output_n_global = output_n
        actual = [row[-1] for row in fold]
        # print(predicted)
        accuracy = accuracy_score(actual, predicted)
        precision = precision_score(actual, predicted, labels=[4, 5, 3, 2, 1], average='micro')
        recall = recall_score(actual, predicted, average='micro')
        f1score = f1_score(actual, predicted, average='micro')
        # my_formatter.format(accuracy)
        scores.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1score)
    return scores, precisions, recalls, f1scores


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


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            # try:
            #     expected[row[-1]] = 1
            # except:
            #     new_r = (row[-1]-1)
            #     expected[new_r] = 1
            # try:
            #     expected = [0 for i in range(n_outputs)]
            #     expected[row[-1]] = 1
            # except:
            #     n_outputs=n_outputs+1
            #     expected = [0 for i in range(n_outputs)]
            #     expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(n_global, train, test, l_rate, n_epoch, n_hidden):
    global list_weight
    n_inputs = len(train[0]) - 1
    # print(n_inputs)
    n_outputs = len(set([row[-1] for row in train]))
    # print(n_outputs)
    if n_global != 0 and n_outputs != n_global:
        n_outputs = n_global
        # print('setelah perubahan: ', n_outputs)
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    # print(network)
    list_weight.append(network)
    for row in test:
        prediction = predict(network, row)
        # print(prediction)
        predictions.append(prediction)
    return (predictions), n_outputs


def mainTraining(filenya):
    start_time = time.time()
    seed(1)
    # load and prepare data
    # filenya = 'csv_data/2018_datasiap_dataawalbanget.csv'
    # print(filenya)
    # filename = filenya
    dataset = load_csv('uploads/'+filenya)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    global list_weight
    list_weight.clear()
    # print(dataset)
    # print(minmax)
    # evaluate algorithm
    n_folds = 10
    l_rate = 0.25
    n_epoch = 100
    n_hidden = 3
    scores, precisions, recalls, f1scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % scores)
    sum_score = (sum(scores) / float(len(scores)))
    print('Mean Accuracy: %.3f%%' % sum_score)
    print('---------------------------')
    print('Precisions: %s' % precisions)
    sum_precisions = (sum(precisions) / float(len(precisions)))
    print('Mean Precision: %.3f%%' % sum_precisions)
    print('---------------------------')
    print('Recalls: %s' % recalls)
    sum_recalls = (sum(recalls) / float(len(recalls)))
    print('Mean Recall: %.3f%%' % sum_recalls)
    print('---------------------------')
    print('F-Measure: %s' % f1scores)
    sum_f1scores = (sum(f1scores) / float(len(f1scores)))
    print('Mean F-Measure: %.3f%%' % sum_f1scores)
    print('---------------------------')
    time_end = int(time.time() - start_time)
    print("--- %s seconds ---" % time_end)
    max_of_list = max(scores)
    max_index_list = scores.index(max_of_list)
    best_weight = list_weight[max_index_list]
    # print(best_weight)
    print()
    return sum_score, sum_precisions, sum_recalls, sum_f1scores, \
           time_end, scores, precisions, recalls, f1scores, best_weight, minmax


# mainTraining('uploads/clean_data.csv')

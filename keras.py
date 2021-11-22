# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import csv
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
#
#
data = pd.read_csv('hasil/2018_template_perhitungan.csv', quoting=csv.QUOTE_NONE)
# X = data[['Timestamp', 'Lintang', 'Bujur']]
# y = data[['Kedalaman(KM)', 'Kelas']]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(data)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)
#
#
# def create_model(neurons, activation, optimizer, loss):
#     model = Sequential()
#     model.add(Dense(neurons, activation=activation, input_shape=(3,)))
#     model.add(Dense(neurons, activation=activation))
#     model.add(Dense(2, activation='softmax'))
#
#     model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
#
#     return model
#
# model = KerasClassifier(build_fn=create_model, verbose=0)
#
# # neurons = [16, 64, 128, 256]
# neurons = [16]
# # batch_size = [10, 20, 50, 100]
# batch_size = [10]
# epochs = [10]
# # activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'exponential']
# activation = ['sigmoid', 'relu']
# # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# optimizer = ['SGD', 'Adadelta']
# loss = ['squared_hinge']
#
# param_grid = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, activation=activation, optimizer=optimizer, loss=loss)
#
# model = Sequential()
# model.add(Dense(16, activation='relu', input_shape=(3,)))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(2, activation='softmax'))
#
# model.compile(optimizer='SGD', loss='squared_hinge', metrics=['accuracy'])
#
# model.fit(X_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data=(X_test, y_test))
#
# [test_loss, test_acc] = model.evaluate(X_test, y_test)
# print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


# import platform
# print(platform.architecture())

# data = data.drop(labels=range(150, 381), axis=0)

from random import seed
from csv import reader
import pandas as pd
import csv
import json


# def load_csv(filename):
#     dataset = list()
#     with open(filename, 'r') as file:
#         csv_reader = reader(file)
#         next(csv_reader)
#         for row in csv_reader:
#             if not row:
#                 continue
#             dataset.append(row)
#     return dataset
#
# def str_column_to_float(dataset, column):
#     for row in dataset:
#         row[column] = float(row[column].strip())
#
#
#
# def main():
#     # Test
#     seed(1)
#     # load and prepare data
#     filename1 = 'hasil/2018_mapping.csv'
#     dataset1 = load_csv(filename1)
#     for i in range(len(dataset1[0]) - 1):
#         str_column_to_float(dataset1, i)
#     print(dataset1)
#     x = map(json.dumps, dataset1)
#     print(x)
#
#
# main()


# df = pd.read_csv('hasil/2018_datatesting.csv', quoting=csv.QUOTE_NONE)
# print(df)
# x= df[['Lintang','Bujur','Kelas']]
# print(x)
# x = x.to_dict('index')
# print(x)
# x.to_csv('hasil/2018_mapping.csv', encoding='utf-8', index=False)

df = pd.read_csv('hasil/2018_datafix.csv', quoting=csv.QUOTE_NONE)
df.drop(df[df['Magnitudo(SR)'] < 1.3].index, inplace=True)
df = df.reset_index(drop=True)
# print(df)
x= df[['Tanggal', 'Waktu(UTC)', 'Lintang', 'Bujur', 'Kedalaman(KM)', 'Magnitudo(SR)']]
# print(x)
# x = x.to_dict('index')
# print(x)
x.to_csv('hasil/2018_dataawalbanget.csv', encoding='utf-8', index=False)



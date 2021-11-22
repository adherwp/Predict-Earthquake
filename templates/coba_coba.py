import time
import datetime
import numpy as np
import csv
import pandas as pd
import json

# ['00:34:30', '02:02:02', '02:02:03', '02:02:04', '02:02:05', '02:02:07', '02:02:08', '02:02:09', '02:02:10', '02:02:11', '02:02:12', '12:12:13', '12:12:14', '12:12:15'] ['3.9', '3.9', '3.2', '3.3', '3.5', '3.6', '2.7', '2.2', '2.9', '3.0', '3.1', '3.2', '3.3', '3.4']
# ['00:43:05', '00:52:32', '02:47:53', '03:03:26', '03:07:09', '03:24:53', '03:32:13', '04:30:53', '04:59:02', '05:22:39', '06:22:42', '07:26:01', '07:31:59', '07:48:02'] [2.2, 2.6, 2.4, 2.7, 2.3, 4.5, 2.8, 3.4, 3.4, 2.2, 2.5, 4.0, 2.9, 2.4]
# tanggal = '2018-01-01'
# waktu = ['00:34:30', '02:02:02', '02:02:03', '02:02:04', '02:02:05', '02:02:07', '02:02:08', '02:02:09', '02:02:10', '02:02:11', '02:02:12', '12:12:13', '12:12:14', '12:12:15']
# datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
# print(waktu)
# for i in range(len(waktu)):
#     waktu[i] = datetime.datetime.strptime(waktu[i], '%H:%M:%S')
# list_no = ['3.9', '3.9', '3.2', '3.3', '3.5', '3.6', '2.7', '2.2', '2.9', '3.0', '3.1', '3.2', '3.3', '3.4']
# print(list_no)
# list_no = list(np.float_(list_no))
# print(list_no)

# df_template = pd.read_csv('../uploads/template_perhitungan.csv', quoting=csv.QUOTE_NONE)
# time, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass = \
#             df_template['time'].tolist(), df_template['mean_magnitude'].tolist(), df_template['energy'].tolist(), \
#             df_template['b-value'].tolist(), df_template['mean_square'].tolist(), df_template['max_difference'].tolist(), \
#             df_template['koef_variation'].tolist(), df_template['freq_gempa'].tolist(), df_template['class'].tolist()
# print(time)

# s1 = '22:22:22'
# s2 = '23:22'
# FMT = '%H:%M:%S'
# FMT2 = '%H:%M'
# tdate = datetime.datetime.strptime(s2, FMT2) - datetime.datetime.strptime(s1, FMT)
# print(tdate)

df = pd.read_csv('../csv_data/2018_dataawalbanget.csv', quoting=csv.QUOTE_NONE)
# print(df)

df_new = df[:2800]

print(df_new)

df_new.to_csv('potongan.csv', encoding='utf-8', index=False)

# row=[0.2214765100671141, 0.5527619746742521, 0.004054054054054054, 0.978688593523626, 0]
# print(row[:-1])
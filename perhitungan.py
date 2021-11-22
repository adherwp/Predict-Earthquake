import csv
import pandas as pd
import math
from datetime import datetime
from numpy import mean
from sklearn.cluster import KMeans


def hitungDelapan(filenya):
    df = pd.read_csv('uploads/'+filenya, quoting=csv.QUOTE_NONE)
    # df.drop(df[df['Magnitudo(SR)'] < 1.3].index, inplace=True)
    df = df.reset_index(drop=True)
    # print(df)

    # create an Empty DataFrame object With column names only
    df_perhitungan = pd.DataFrame(columns=['time', 'mean_magnitude', 'energy', 'b-value', 'mean_square', 'max_difference', 'koef_variation', 'freq_gempa'])
    # print(df_perhitungan)

    def power(my_list):
        return [x**2 for x in my_list]

    i = 0
    while i < len(df.index):
        time_list = []
        magnitudo_list = []
        x = i
        batas=(i+14)
        while x < batas:
            time_list.append(df['Waktu(UTC)'].values[x])
            magnitudo_list.append(df['Magnitudo(SR)'].values[x])
            x = x+1

        # print(time_list)
        # print(magnitudo_list)
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
            s1 = df['Waktu(UTC)'].values[max_index - 1]
            s2 = df['Waktu(UTC)'].values[max_index]
            FMT = '%H:%M:%S'
            tdate = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
            tdelta = tdate.total_seconds()/60
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

        df_perhitungan = df_perhitungan.append({'time': tdelta, 'mean_magnitude': mean_pop, 'energy': energy, 'b-value': b_value, 'mean_square': mean_square, 'max_difference': max_diff, 'koef_variation': koef, 'freq_gempa': kenaikan}, ignore_index=True)
        i = i+14

    # pd.set_option('display.max_columns', None)
    # print(df_perhitungan)
    data_points = df_perhitungan.iloc[:, 0:8]
    kmeans = KMeans(init="random", n_clusters=5, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(data_points)
    predict = kmeans.predict(data_points)
    df_perhitungan["cluster"] = predict
    mynames = {"0": "1", "1": "2", "2": "3", "3": "4", "4": "5"}
    df_perhitungan["class"] = [mynames[str(i)] for i in df_perhitungan.cluster]
    df_perhitungan = df_perhitungan.drop(columns=['cluster'])
    # print(df_perhitungan)
    df_perhitungan.to_csv('uploads/template_perhitungan.csv', encoding='utf-8', index=False)
    # print(df_perhitungan['class'].value_counts())
    # print(df_perhitungan['cluster'].value_counts())


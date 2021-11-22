import csv
import pandas as pd
import math
from datetime import datetime
from numpy import mean

df = pd.read_csv('hasil/2018_no_outlier.csv', quoting=csv.QUOTE_NONE)

df.drop(df[df['Magnitudo(SR)'] < 1.3].index, inplace=True)
print(df)

def power(my_list):
    return [ x**2 for x in my_list ]

# create an Empty DataFrame
# object With column names only
# df_perhitungan = pd.DataFrame(columns=['time', 'Mman_magnitude', 'energy', 'b-value', 'mean_square_deviation', 'max_difference', 'koef_variaton', 'freq_gempa'])
# print(df_perhitungan)

time_list=[]
magnitudo_list=[]
for i in range(14):
    if i==0:
        time_list = [df['Waktu(UTC)'].values[i]]
        magnitudo_list = [df['Magnitudo(SR)'].values[i]]
    else:
        time_list.append(df['Waktu(UTC)'].values[i])
        magnitudo_list.append(df['Magnitudo(SR)'].values[i])

max_value = max(magnitudo_list)
min_value = min(magnitudo_list)
max_index = magnitudo_list.index(max_value)

magnitudo_list_pop = magnitudo_list[:]
magnitudo_list_pop.pop(max_index)

# proses no 1
s1=df['Waktu(UTC)'].values[max_index-1]
s2=df['Waktu(UTC)'].values[max_index]
FMT = '%H:%M:%S'
tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)

# proses no 2
mean_all=mean(magnitudo_list)
mean_pop=mean(magnitudo_list_pop)

# proses no 3
Sum = sum(magnitudo_list_pop)
Pangkat = 4.8 + (1.5*max_value)
Pow = math.pow(Sum, Pangkat)
energy = (math.sqrt(Pow))/1000000000

# proses no 4
b_value = 0.4342944819/(mean_all-mean_pop)

# proses no 5
kurung = math.pow((max_value-mean_pop), 2)
totalnya_mean = (1/14)*kurung
mean_square = math.sqrt(totalnya_mean)

# proses no 6
max_diff = max_value-min_value

#proses no 7
magnitudo_list_pop_pow = power(magnitudo_list_pop)
kurung_awal = (1/13)*sum(magnitudo_list_pop_pow)
kurung_kedua = math.pow(mean_pop, 2)
akar_koef = math.sqrt((kurung_awal-kurung_kedua))
koef = akar_koef/mean_pop

# proses no 8
slices_list = magnitudo_list[:max_index+1][:]
kenaikan=0
for i in range(len(slices_list)):
    if i is not len(slices_list)-1:
        if slices_list[i]<slices_list[i+1]:
            kenaikan=kenaikan+1

# print(time_list)
print(kenaikan)
print(magnitudo_list)
# print(magnitudo_list_pop)
# print(magnitudo_list_pop_pow)
# print(mean_all)
# no 1
print(tdelta)
# no 2
print(mean_pop)
# no 3
print(energy)
# no 4
print(b_value)
# no 5
print(mean_square)
# no 6
print(max_diff)
# no 7
print(koef)
# no 8
print()
import csv
import pandas as pd
import math
import time
import datetime
from numpy import mean


def convertData(filenya):
    df = pd.read_csv('uploads/'+filenya, quoting=csv.QUOTE_NONE)
    df = df.reset_index(drop=True)
    # print(df)

    # Tanggal,Waktu(UTC),Lintang,Bujur,Kedalaman(KM),Magnitudo(SR),TypeMagnitudo,smaj,smin,az,rms,cPhase,Letak
    # data = df[['Tanggal', 'Waktu(UTC)', 'Lintang', 'Bujur', 'Kedalaman(KM)', 'Magnitudo(SR)']]
    # print(data)
    df["Magnitudo(SR)"] = pd.to_numeric(df["Magnitudo(SR)"], downcast="float")

    timestamp = []
    for d, t in zip(df['Tanggal'], df['Waktu(UTC)']):
        try:
            ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
            timestamp.append(time.mktime(ts.timetuple()))
        except ValueError:
            # print('ValueError')
            timestamp.append('ValueError')

    timeStamp = pd.Series(timestamp)
    # print(timeStamp)
    # data['Timestamp'] = timeStamp

    frame = {'Timestamp': timeStamp}
    result = pd.DataFrame(frame)
    # print(result)

    drop_data = df.drop(['Tanggal', 'Waktu(UTC)'], axis=1)
    list_data = [drop_data, result]
    final_data = pd.concat(list_data, axis=1)
    # print(final_data)

    def myfunc(mg):
        if mg < 3.5:
            mg = 1
        elif mg < 4.0:
            mg = 2
        elif mg < 4.5:
            mg = 3
        elif mg < 5.0:
            mg = 4
        else:
            mg = 5
        return mg

    final_data['Kelas'] = final_data.apply(lambda x: myfunc(x['Magnitudo(SR)']), axis=1)
    # df['Kelas'] = df.apply(myfunc(df['Magnitudo(SR)']))
    final_data = final_data.drop(['Magnitudo(SR)'], axis=1)
    # print(final_data)

    final_data.to_csv('uploads/clean_data.csv', encoding='utf-8', index=False)

    x = final_data[['Lintang','Bujur','Kelas']]
    # x = x.to_dict('index')
    x.to_csv('uploads/mapping.csv', encoding='utf-8', index=False)


# df = pd.read_csv('hasil/2018_datasiap.csv', quoting=csv.QUOTE_NONE)
# print(df)

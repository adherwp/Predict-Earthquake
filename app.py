from flask_bootstrap import Bootstrap
import os
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import csv
import time
import datetime
import numpy as np
from pathlib import Path
from perhitungan import hitungDelapan
from convert_data_v1 import convertData
from model import mainTraining
from predict import main_predict
from predict_t import main_predict_t


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
filename_globalnya = ''
score, precision, recall, f1score, time_estimate = 0.0, 0.0, 0.0, 0.0, 0.0
score_t, precision_t, recall_t, f1score_t, time_estimate_t = 0.0, 0.0, 0.0, 0.0, 0.0
scores, precisions, recalls, f1scores = list(), list(), list(), list()
scores_t, precisions_t, recalls_t, f1scores_t = list(), list(), list(), list()
status_training = False
best_w, best_w_t = list(), list()
minmax_w, minmax_w_t = list(), list()
l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo = list(), list(), list(), list(), list(), list()
l_timestamp, l_kelas = list(), list()
timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass = list(), list(), list(), list(), list(), list(), list(), list(), list()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app():
  app = Flask(__name__)
  app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  Bootstrap(app)
  return app


app = create_app()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/visualisasi')
def visualisasi():
    my_file = Path("uploads/mapping.csv")
    if my_file.is_file():
        df = pd.read_csv('uploads/mapping.csv', quoting=csv.QUOTE_NONE)
        dict = df.to_dict('index')
        return render_template('visualizing.html', data=dict)
    else:
        return render_template('no_visualizing.html')


@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            error = "No File Part"
            return render_template('training.html', error=error)
        if file.filename == '':
            error = "No Selected File"
            return render_template('training.html', error=error)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv'))
            hitungDelapan('data.csv')
            convertData('data.csv')
            global filename_globalnya, status_training, best_w, best_w_t, minmax_w, minmax_w_t
            filename_globalnya = filename
            global score, precision, recall, f1score, time_estimate, scores, precisions, recalls, f1scores
            global score_t, precision_t, recall_t, f1score_t, time_estimate_t, scores_t, precisions_t, recalls_t, f1scores_t
            score, precision, recall, f1score, time_estimate, \
            scores, precisions, recalls, f1scores, best_w, minmax_w = mainTraining('clean_data.csv')
            score_t, precision_t, recall_t, f1score_t, time_estimate_t, \
            scores_t, precisions_t, recalls_t, f1scores_t, best_w_t, minmax_w_t = mainTraining('template_perhitungan.csv')
            status_training = True
            # flash(filename)
            global l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo, l_timestamp, l_kelas
            global timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass
            df_awal = pd.read_csv('uploads/data.csv', quoting=csv.QUOTE_NONE)
            df_clean = pd.read_csv('uploads/clean_data.csv', quoting=csv.QUOTE_NONE)
            df_template = pd.read_csv('uploads/template_perhitungan.csv', quoting=csv.QUOTE_NONE)
            # dict = df_awal.to_dict()
            l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo, l_timestamp, l_kelas = df_awal['Tanggal'].tolist(), df_awal['Waktu(UTC)'].tolist(), df_awal['Lintang'].tolist(),\
                                                                               df_awal['Bujur'].tolist(), df_awal['Kedalaman(KM)'].tolist(), df_awal['Magnitudo(SR)'].tolist(),\
                                                                               df_clean['Timestamp'].tolist(), df_clean['Kelas'].tolist()
            timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass = df_template['time'].tolist(), df_template['mean_magnitude'].tolist(), df_template['energy'].tolist(),\
                                                                               df_template['b-value'].tolist(), df_template['mean_square'].tolist(), df_template['max_difference'].tolist(),\
                                                                               df_template['koef_variation'].tolist(), df_template['freq_gempa'].tolist(), df_template['class'].tolist()
            return render_template('training_result.html',
                                   file=filename_globalnya,
                                   score=score, precision=precision, recall=recall, f1score=f1score, time=time_estimate,
                                   score_t=score_t, precision_t=precision_t, recall_t=recall_t, f1score_t=f1score_t, time_t=time_estimate_t,
                                   scores=scores, precisions=precisions, recalls=recalls, f1scores=f1scores,
                                   scores_t=scores_t, precisions_t=precisions_t, recalls_t=recalls_t, f1scores_t=f1scores_t,
                                   data_list=zip(l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo),
                                   clean_list=zip(l_timestamp, l_kelas, l_lintang, l_bujur, l_kedalaman),
                                   data_list2=zip(l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo),
                                   clean_list2=zip(timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass))
            # return redirect(url_for('training_result'))
    else:
        my_file = Path("uploads/data.csv")
        if my_file.is_file() and filename_globalnya != '':
            return render_template('training_result.html',
                                   file=filename_globalnya,
                                   score=score, precision=precision, recall=recall, f1score=f1score, time=time_estimate,
                                   score_t=score_t, precision_t=precision_t, recall_t=recall_t, f1score_t=f1score_t, time_t=time_estimate_t,
                                   scores=scores, precisions=precisions, recalls=recalls, f1scores=f1scores,
                                   scores_t=scores_t, precisions_t=precisions_t, recalls_t=recalls_t, f1scores_t=f1scores_t,
                                   data_list=zip(l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo),
                                   clean_list=zip(l_timestamp, l_kelas, l_lintang, l_bujur, l_kedalaman),
                                   data_list2=zip(l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo),
                                   clean_list2=zip(timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass))
        elif my_file.is_file() and filename_globalnya == '':
            filename_globalnya = 'data terakhir yang digunakan'
            score, precision, recall, f1score, time_estimate, \
            scores, precisions, recalls, f1scores, best_w, minmax_w = mainTraining('clean_data.csv')
            score_t, precision_t, recall_t, f1score_t, time_estimate_t, \
            scores_t, precisions_t, recalls_t, f1scores_t, best_w_t, minmax_w_t = mainTraining('template_perhitungan.csv')
            status_training = True
            # flash(filename)
            df_awal = pd.read_csv('uploads/data.csv', quoting=csv.QUOTE_NONE)
            df_clean = pd.read_csv('uploads/clean_data.csv', quoting=csv.QUOTE_NONE)
            df_template = pd.read_csv('uploads/template_perhitungan.csv', quoting=csv.QUOTE_NONE)
            l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo, l_timestamp, l_kelas = df_awal['Tanggal'].tolist(), df_awal['Waktu(UTC)'].tolist(), df_awal['Lintang'].tolist(), \
                                                                               df_awal['Bujur'].tolist(), df_awal['Kedalaman(KM)'].tolist(), df_awal['Magnitudo(SR)'].tolist(),\
                                                                               df_clean['Timestamp'].tolist(), df_clean['Kelas'].tolist()
            timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass = \
            df_template['time'].tolist(), df_template['mean_magnitude'].tolist(), df_template['energy'].tolist(), \
            df_template['b-value'].tolist(), df_template['mean_square'].tolist(), df_template['max_difference'].tolist(), \
            df_template['koef_variation'].tolist(), df_template['freq_gempa'].tolist(), df_template['class'].tolist()
            return render_template('training_result.html',
                                   file=filename_globalnya,
                                   score=score, precision=precision, recall=recall, f1score=f1score, time=time_estimate,
                                   score_t=score_t, precision_t=precision_t, recall_t=recall_t, f1score_t=f1score_t,
                                   time_t=time_estimate_t,
                                   scores=scores, precisions=precisions, recalls=recalls, f1scores=f1scores,
                                   scores_t=scores_t, precisions_t=precisions_t, recalls_t=recalls_t,
                                   f1scores_t=f1scores_t,
                                   data_list=zip(l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo),
                                   clean_list=zip(l_timestamp, l_kelas, l_lintang, l_bujur, l_kedalaman),
                                   data_list2=zip(l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo),
                                   clean_list2=zip(timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass))
        else:
            return render_template('training.html')


@app.route('/delete')
def delete():
    os.remove("uploads/data.csv")
    os.remove("uploads/clean_data.csv")
    os.remove("uploads/template_perhitungan.csv")
    os.remove("uploads/mapping.csv")
    global filename_globalnya, status_training, best_w, best_w_t, minmax_w, minmax_w_t
    global score, precision, recall, f1score, time_estimate, scores, precisions, recalls, f1scores
    global score_t, precision_t, recall_t, f1score_t, time_estimate_t, scores_t, precisions_t, recalls_t, f1scores_t
    global l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo, l_timestamp, l_kelas
    global timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass
    filename_globalnya = ''
    score, precision, recall, f1score, time_estimate = 0.0, 0.0, 0.0, 0.0, 0.0
    score_t, precision_t, recall_t, f1score_t, time_estimate_t = 0.0, 0.0, 0.0, 0.0, 0.0
    scores, precisions, recalls, f1scores = list(), list(), list(), list()
    scores_t, precisions_t, recalls_t, f1scores_t = list(), list(), list(), list()
    status_training = False
    best_w, best_w_t = list(), list()
    minmax_w, minmax_w_t = list(), list()
    l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo = list(), list(), list(), list(), list(), list()
    l_timestamp, l_kelas = list(), list()
    timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass = list(), list(), list(), list(), list(), list(), list(), list(), list()
    return redirect(url_for('training'))


@app.route('/re_train')
def re_train():
    global filename_globalnya, status_training, best_w, best_w_t, minmax_w, minmax_w_t
    global score, precision, recall, f1score, time_estimate, scores, precisions, recalls, f1scores
    global score_t, precision_t, recall_t, f1score_t, time_estimate_t, scores_t, precisions_t, recalls_t, f1scores_t
    global l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo, l_timestamp, l_kelas
    global timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass
    filename_globalnya = ''
    score, precision, recall, f1score, time_estimate = 0.0, 0.0, 0.0, 0.0, 0.0
    score_t, precision_t, recall_t, f1score_t, time_estimate_t = 0.0, 0.0, 0.0, 0.0, 0.0
    scores, precisions, recalls, f1scores = list(), list(), list(), list()
    scores_t, precisions_t, recalls_t, f1scores_t = list(), list(), list(), list()
    status_training = False
    best_w, best_w_t = list(), list()
    minmax_w, minmax_w_t = list(), list()
    l_tanggal, l_waktu, l_lintang, l_bujur, l_kedalaman, l_magnitudo = list(), list(), list(), list(), list(), list()
    l_timestamp, l_kelas = list(), list()
    timee, mean_magnitude, energy, b_value, mean_square, max_difference, koef_variation, freq_gempa, klass = list(), list(), list(), list(), list(), list(), list(), list(), list()
    return redirect(url_for('training'))

@app.route('/testing', methods=['GET', 'POST'])
def testing():
    print(minmax_w)
    print(best_w)
    if request.method == 'POST':
        tanggal = request.form["tanggal"]
        waktu = request.form["time"]
        try:
            date_time = datetime.datetime.strptime(tanggal + ' ' + waktu, '%Y-%m-%d %H:%M:%S')
        except:
            date_time = datetime.datetime.strptime(tanggal + ' ' + waktu, '%Y-%m-%d %H:%M')
        lintang = request.form["lintang"]
        bujur = request.form["bujur"]
        kedalaman = request.form["kedalaman"]
        # print(date_time, lintang, bujur, kedalaman, best_w, minmax_w)
        prediksi = main_predict(date_time, lintang, bujur, kedalaman, best_w, minmax_w)
        prediksi = prediksi+1
        print(prediksi)
        # print()
        return render_template('testing_result.html',
                               result=int(prediksi),
                               lat=float(lintang), long=float(bujur), deep=float(kedalaman))
    else:
        my_file = Path("uploads/clean_data.csv")
        if my_file.is_file() and status_training is True:
            return render_template('testing.html')
        else:
            return render_template('no_testing.html')


@app.route('/testing_t', methods=['GET', 'POST'])
def testing_t():
    print(minmax_w_t)
    print(best_w_t)
    if request.method == 'POST':
        tanggal = request.form["date"]
        waktu = request.form.getlist('time[]')
        magnitudo = request.form.getlist('magnitudo[]')
        magnitudo = list(np.float_(magnitudo))
        # print(waktu, magnitudo)
        # print(best_w_t, minmax_w_t)
        prediksi = main_predict_t(tanggal, waktu, magnitudo, best_w_t, minmax_w_t)
        prediksi = prediksi + 1
        print(prediksi)
        # print()
        return render_template('testing_t_result.html',
                               result=int(prediksi),
                               review_list=zip(waktu, magnitudo),
                               date=tanggal)
    else:
        my_file = Path("uploads/template_perhitungan.csv")
        if my_file.is_file() and status_training is True:
            return render_template('testing_t.html')
        else:
            return render_template('no_testing_t.html')


if __name__ == '__main__':
    app.run()

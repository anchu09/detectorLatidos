#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:27:37 2022

@author: dani
"""

import glob
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Custom_Data_Generator import CustomDataGenerator
import models
import utils

os.chdir(os.getcwd()[:-len("heartbeat_detector")])

path = './semi_preprocessed_signals/original_ecg'
files = glob.glob(path + "/*.csv")
diccionarioDatos = {}
for filename in files:
    diccionarioDatos[filename[(len(path) + 1):len(filename) - 4]] = pd.read_csv(filename, index_col=None,
                                                                                on_bad_lines='skip', delimiter="\t",
                                                                                header=None)

datasetCustom = pd.read_csv('./sliced_signals/oneHotTrainingResumido.csv', index_col=None, on_bad_lines='skip', sep='\s+', header=None)
datasetCustom.iloc[:, 0] = np.array([utils.to_seconds(t) for t in datasetCustom.iloc[:, 0]])

labelencoder = LabelEncoder()
datasetCustom.iloc[:, 2] = labelencoder.fit_transform(datasetCustom.iloc[:, 2])
transformer = ColumnTransformer(transformers=[("ecg", OneHotEncoder(categories='auto'), [2])], remainder='passthrough')
datasetCustom = transformer.fit_transform(datasetCustom)

numeros_posibles = sorted(list(diccionarioDatos.keys()))

anotaciones_ficheros = {'100': 'S', '101': 'S', '102': 'Q', '103': 'S', '104': 'Q', '105': 'Q', '106': 'V', '107': 'Q',
                        '108': 'F', '109': 'F', '111': 'V', '112': 'S', '113': 'S', '114': 'F', '115': 'N', '116': 'S',
                        '117': 'S', '118': 'S', '119': 'V', '121': 'S', '122': 'N', '123': 'V', '124': 'F', '200': 'F',
                        '201': 'F', '202': 'F', '203': 'F', '205': 'F', '207': 'S', '208': 'F', '209': 'S', '210': 'F',
                        '212': 'N', '213': 'F', '214': 'F', '215': 'F', '217': 'Q', '219': 'F', '220': 'S', '221': 'V',
                        '222': 'S', '223': 'F', '228': 'S', '230': 'V', '231': 'S', '232': 'S', '233': 'F', '234': 'S'}
label_per_file = list(anotaciones_ficheros.values())
factor = int(len(numeros_posibles) * 0.3)

x_train, x_test = train_test_split(numeros_posibles, test_size=factor, stratify=label_per_file)

x_train = sorted(x_train)
x_test = sorted(x_test)

path_x = "./sliced_signals/ecgs/"
path_y = "./sliced_signals/annotations/"

lista_paths_test_y = []
lista_paths_test_x = []

for numero_test in x_test:
    filesx = glob.glob(path_x + str(numero_test) + "/*.csv")
    lista_paths_test_x.extend(filesx)
    filesy = glob.glob(path_y + str(numero_test) + "/*.csv")
    lista_paths_test_y.extend(filesy)

lista_paths_train_y = []
lista_paths_train_x = []
for numero_train in x_train:
    filesx2 = glob.glob(path_x + str(numero_train) + "/*.csv")
    lista_paths_train_x.extend(filesx2)
    filesy2 = glob.glob(path_y + str(numero_train) + "/*.csv")
    lista_paths_train_y.extend(filesy2)

lista_paths_test_y = sorted(lista_paths_test_y)
lista_paths_test_x = sorted(lista_paths_test_x)
lista_paths_train_y = sorted(lista_paths_train_y)
lista_paths_train_x = sorted(lista_paths_train_x)


train = CustomDataGenerator(lista_paths_train_x, lista_paths_train_y, batch_size=1, train=True)
test = CustomDataGenerator(lista_paths_test_x, lista_paths_test_y, batch_size=1, train=False)



def train_data_generator():
    for i in range(len(train)):
        x, y = train[i]
        yield x[0], y[0]

def test_data_generator():
    for i in range(len(test)):
        x, y = test[i]
        yield x[0], y[0]

datasettrain = tf.data.Dataset.from_generator(train_data_generator,
                                              output_signature=(tf.TensorSpec(shape=(5000, 2), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(5000, 6), dtype=tf.int32)))

datasettest = tf.data.Dataset.from_generator(test_data_generator,
                                             output_signature=(tf.TensorSpec(shape=(5000, 2), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(5000, 6), dtype=tf.int32)))
input_shape = (5000, 2)
model = models.model_build_func(input_shape)

datasettrain = datasettrain.shuffle(len(train)).batch(1)
datasettest = datasettest.batch(1)

history = model.fit_generator(datasettrain, epochs=1, validation_data=datasettest)
train_loss = history.history['loss']
test_loss = history.history['val_loss']

test_predictions = model.predict(datasettest)

L = 51
b = utils.define_ma(L)

for i in np.arange(len(test_predictions)):
    for k in np.arange(test_predictions[i].shape[1]):
        test_predictions[i][:, k] = sig.filtfilt(b, 1, test_predictions[i][:, k])

max_index = tf.argmax(test_predictions, axis=-1)
one_hot_output = tf.one_hot(max_index, depth=6)
one_hot_output = one_hot_output.numpy()

# for i in np.arange(len(lista_paths_test_x)):
#
#     ecg_actual = np.asarray(np.loadtxt(lista_paths_test_x[i])).astype(np.float32)
#     decoder_string = "F = 0\nN = 1\nQ = 2\nS = 3\nV = 4\nZ = 5"
#     plt.plot(ecg_actual[:, 0] + 1, label=decoder_string)
#     plt.title(str(i) + ": " + lista_paths_test_x[i][29:-4])
#     etiquetasactuales = np.asarray(np.loadtxt(lista_paths_test_y[i], delimiter=',')).astype(np.float32)
#     for k in np.arange(5):
#         plt.plot(np.arange(len(one_hot_output[i])), one_hot_output[i][:, k], label="PRED: " + str(k))
#         plt.plot(etiquetasactuales[:, k] - 0.05 - (k / 100), label="ORIG: " + str(k))
#
#     plt.plot(np.arange(len(one_hot_output[i])), one_hot_output[i][:, 5] - 0.4, label="PRED: " + str(5), color="brown")
#     plt.plot(etiquetasactuales[:, 5] - 0.45, label="ORIG: " + str(5), color="black")
#     plt.ylim(0.25, )
#
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.subplots_adjust(right=0.75)
#
#     os.makedirs("results/plots", exist_ok=True)
#
#     plt.savefig("results/plots/" + str(i) + ": " + lista_paths_test_x[i][29:-4] + ".png")
#     #
#     plt.show()
#     plt.figure()
# #


decoded_labels = []
for i in range(one_hot_output.shape[0]):
    print(str(i)+"/"+str(one_hot_output.shape[0]))

    for j in range(one_hot_output.shape[1]):
        label_encoded = np.argmax(one_hot_output[i, j, :])
        label_decoded = labelencoder.inverse_transform([label_encoded])[0]
        decoded_labels.append(label_decoded)

decoded_labels = np.transpose(np.asarray(decoded_labels).reshape(one_hot_output.shape[0], one_hot_output.shape[1]))

rg_min = 0
rg_max = 130
matriz_nueva = np.empty((650000, factor), dtype='U1')
for i in np.arange(factor):
    fila_actual = np.empty(0)
    contdor = 0
    for k in np.arange(rg_min, rg_max):
        contdor += 1
        fila_actual = np.concatenate((fila_actual, decoded_labels[:, int(k)]))
    rg_min = rg_max
    rg_max = rg_max + 130
    matriz_nueva[:, i] = fila_actual

lista_archivos_resultado = set([elem[len(lista_paths_test_y[0])-19:len(lista_paths_test_y[0])-16] for elem in lista_paths_test_y])
fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
carpeta_actual = os.path.join("./results/bxb/", fecha_actual)
os.makedirs(carpeta_actual, exist_ok=True)

frecMuestreo = 360
for col_num in np.arange(len(lista_archivos_resultado)):
    f = open("./results/bxb/" + fecha_actual + "/" + str(sorted(list(lista_archivos_resultado))[col_num]) + ".csv", 'w')
    columna_actual = matriz_nueva[:, col_num]
    matriz = np.column_stack((np.arange(650000), columna_actual))
    for indice in np.arange(650000):
        if matriz[indice, 1] == 'Z':
            continue
        else:
            latido_actual = matriz[indice, 1]
            indice_adelantado = indice
            while (True):
                if indice_adelantado >= 650000:
                    break

                elif latido_actual == matriz[indice_adelantado, 1]:
                    indice_adelantado += 1
                else:
                    break
            ancho_latido = indice_adelantado - indice
            for muestra in np.arange(indice, indice_adelantado):
                if muestra == int((indice + indice_adelantado) / 2):
                    continue
                else:
                    matriz[muestra, 1] = 'Z'
            indice = indice_adelantado
    # anotaciones
    for indice in np.arange(650000):
        if matriz[indice, 1] == 'Z':
            continue
        else:
            tiempo = int(matriz[indice, 0]) / 360
            tiempostr = utils.segundos_a_segundos_minutos_y_horas(tiempo)
            stringEspacios = utils.cadenaEspacios(len(tiempostr))
            stringtiempo = stringEspacios + tiempostr
            string = stringtiempo + '{:9d}'.format(int(matriz[indice, 0])) + '     ' + matriz[
                indice, 1] + '{:5d}{:5d}{:5d}'.format(0, 0, 0) + "\n"
            f.write(string)
    f.close()

ruta_directorio = os.getcwd()
ruta_carpeta = os.path.join(ruta_directorio, "results/bxb/" + fecha_actual)

for fichero_atr in sorted(lista_archivos_resultado):
    ruta_archivo = os.path.join(ruta_directorio+"/original_database", fichero_atr + ".atr")
    ruta_copia = os.path.join(ruta_carpeta, fichero_atr + ".atr")
    shutil.copy(ruta_archivo, ruta_copia)

for fichero_hea in sorted(lista_archivos_resultado):
    ruta_archivo = os.path.join(ruta_directorio+"/original_database", fichero_hea + ".hea")
    ruta_copia = os.path.join(ruta_carpeta, fichero_hea + ".hea")
    shutil.copy(ruta_archivo, ruta_copia)

for fichero_qrs in sorted(lista_archivos_resultado):
    ruta_archivo = os.path.join(ruta_directorio+"/original_database", fichero_qrs + ".qrs")
    ruta_copia = os.path.join(ruta_carpeta, fichero_qrs + ".qrs")
    shutil.copy(ruta_archivo, ruta_copia)

carpeta_actual2 = os.path.join("./results/My_annotations/", fecha_actual)
os.makedirs(carpeta_actual2, exist_ok=True)
for nombrefichero in sorted(lista_archivos_resultado):
    os.system(
        "cat ./results/bxb/" + fecha_actual + "/" + nombrefichero + ".csv | wrann -r ./results/bxb/" + fecha_actual + "/" + nombrefichero + " -a myqrs")

    os.system(
        "rdann -r ./results/bxb/" + fecha_actual + "/" + nombrefichero + " -a myqrs>./results/My_annotations/" + fecha_actual + "/" + nombrefichero + ".csv")
    os.system("echo MY_ANNOTATIONS: >> ./results/bxb/" + fecha_actual + "/resultados_bxb.txt")
    os.system(
        "bxb -r ./results/bxb/" + fecha_actual + "/" + nombrefichero + " -a atr myqrs >> ./results/bxb/" + fecha_actual + "/resultados_bxb.txt")
    os.system("echo GQRS_ANNOTATIONS >> ./results/bxb/" + fecha_actual + "/resultados_bxb.txt")
    os.system(
        "bxb -r ./results/bxb/" + fecha_actual + "/" + nombrefichero + " -a atr qrs >> ./results/bxb/" + fecha_actual + "/resultados_bxb.txt")
    os.system(
        "echo ---------------------------------------------------------------------------------------------------------------- >> ./results/bxb/" + fecha_actual + "/resultados_bxb.txt")

plt.figure()

plt.plot(train_loss, label="train_loss")
plt.plot(test_loss, label="test_loss")
plt.legend()
plt.savefig("./results/bxb/" + fecha_actual + "/loss.png")

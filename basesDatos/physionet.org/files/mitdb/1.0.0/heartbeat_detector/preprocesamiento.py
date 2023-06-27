#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:27:37 2022

@author: dani
"""

import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import random
from tensorflow import keras
from keras.utils import Sequence
import tensorflow as tf
from datetime import datetime
import shutil
from keras.layers import Input, Conv1D, MaxPooling1D, LayerNormalization, Flatten, Dense, Dropout, MultiHeadAttention, \
    GlobalMaxPooling1D, Reshape, UpSampling1D
os.chdir(os.getcwd()[:-len("scientificProject")])

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.debugging.set_log_device_placement(True)
#leemos todos los archivos de la carpeta

#
path= 'leiblesDAT'

files = glob.glob(path + "/*.csv")


diccionarioDatos={}

for filename in files:
    diccionarioDatos[filename[(len(path)+1):len(filename)-4]]=pd.read_csv(filename,index_col=None,on_bad_lines='skip', delimiter="\t", header=None)

#atr    reference beat, rhythm, and signal quality annotations



#leemos anotaciones

path= 'leiblesANN'

files = glob.glob(path + "/*.csv")
diccionarioAnotaciones={}

for filename in files:
    data = []
    with open(filename) as file:
        for line in file:
            try:
                line_data = line.strip().split()
                data.append(line_data)
            except:
                continue
    diccionarioAnotaciones[filename[(len(path)+1):len(filename)-4]] = pd.DataFrame(data)




def segundos_a_segundos_minutos_y_horas(segundos):
    horas = int(segundos / 60 / 60)
    segundos -= horas*60*60
    minutos = int(segundos/60)
    segundos -= minutos*60

    if(segundos>=10):

        secs=str(round(segundos,3)).ljust(6, '0')
    else:
        secs=str(round(segundos,3)).ljust(5, '0')

    return str(minutos)+":"+str(secs).zfill(6)



def cadenaEspacios(tamtiempo):
    cantidadespacios=12-tamtiempo
    cadena=""
    for i in range(cantidadespacios):
        cadena+=" "
    return cadena

def to_seconds(time_str):
    minutes, seconds_and_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_and_milliseconds.split('.')
    return int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def fillzeros():

        #colocaomos todas las lineas que contienen muestras donde es 0
        for key in diccionarioAnotaciones.keys():
            f = open("./anotacionesCompletasCeros/"+key+".csv", "w")
            limite_inferior=0
            print(key)


            for index, fila in diccionarioAnotaciones[key].iterrows():
                # print("la fila es")
                # print(fila)
                # print("la fila de uno es")

                # print(fila[1])
                limite_superior=int(fila[1])
                for i in np.arange(limite_inferior,limite_superior):
                    t=i/360
                    t=np.round(t,4)

                    new_row = np.array([t,i,"Z","0","0","0"])
                    for valor in new_row:
                        f.write(str(valor) + " ")
                    f.write("\n")
                f.write(str(int(fila[1])/360)+ " ")
                for valor in fila[1:-1]:
                    f.write(str(valor) + " ")
                f.write("\n")
                limite_inferior=int(fila[1])+1


            for k in np.arange(limite_superior+1,650000):
                t=k/360
                t=np.round(t,4)
                new_row = np.array([t,k,"Z","0","0","0"])
                for valor in new_row:
                    f.write(str(valor) + " ")
                f.write("\n")

            f.close()

def window():

    path= 'anotacionesCompletasCeros'

    files = glob.glob(path + "/*.csv")

    for filename in files:
        data = []
        with open(filename) as file:
            for line in file:
                try:
                    line_data = line.strip().split()
                    data.append(line_data)
                except:
                    continue
        diccionarioAnotacionesCeros[filename[(len(path)+1):len(filename)-4]] = pd.DataFrame(data)
        window_separado(0.15)
        diccionarioAnotacionesCeros.pop(filename[(len(path)+1):len(filename)-4])

def normalizarX():
    from sklearn.preprocessing import MinMaxScaler

    def normalizarX():
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for key in diccionarioDatos.keys():
            print(key)
            canal1 = diccionarioDatos[key][1]
            canal1_normalizado = scaler.fit_transform(canal1.values.reshape(-1, 1))
            diccionarioDatos[key][1] = canal1_normalizado

            canal2 = diccionarioDatos[key][2]
            canal2_normalizado = scaler.fit_transform(canal2.values.reshape(-1, 1))
            diccionarioDatos[key][2] = canal2_normalizado

            print(len(diccionarioDatos[key]))
            diccionarioDatos[key].iloc[:, 1:3].to_csv(
                "./datosNormalizados/" + key + ".csv", index=False, header=False, sep=' ')




diccionarioAnotacionesCeros={}

def window_separado(bandwidth):
    muestras=bandwidth*360
    muestras=round(muestras)
    for key in diccionarioAnotacionesCeros.keys():
        print(key)


        #ya tenemos el vector de las filas que hay que copiar
        listaMuestras = diccionarioAnotaciones[key].iloc[:, 1].values.astype(int)
        for valor in listaMuestras:
            if valor==0:
                valor=1
            fila_a_copiar = diccionarioAnotacionesCeros[key].loc[valor,diccionarioAnotacionesCeros[key].columns[2:]]
            if valor-muestras<0:
                valor_menos_muestras=0
            else:
                valor_menos_muestras=valor-muestras

            if valor+muestras>650000:
                valor_mas_muestras=650000

            else:
                valor_mas_muestras=valor+muestras

            for muestraVentana in np.arange(valor_menos_muestras,valor_mas_muestras):


                diccionarioAnotacionesCeros[key].loc[muestraVentana,diccionarioAnotacionesCeros[key].columns[2:]]=fila_a_copiar

        diccionarioAnotacionesCeros[key].iloc[:,:].to_csv("./con_window_separado/"+key+".csv", index=False, header=False,sep=' ')





# fillzeros()

# window()

# normalizarX()


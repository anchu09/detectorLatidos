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
# from keras.integration_test.models.translation import TransformerDecoder
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import random
from tensorflow import keras
from keras.utils import Sequence
import tensorflow as tf
from datetime import datetime
import shutil
import keras_nlp
from keras_nlp.layers import SinePositionEncoding, TransformerDecoder,TransformerEncoder

from keras.layers import Input, Conv1D, MaxPooling1D, LayerNormalization, Flatten, Dense, Dropout, MultiHeadAttention, \
    GlobalMaxPooling1D, Reshape, UpSampling1D, Conv1DTranspose

os.chdir(os.getcwd()[:-len("scientificProject")])

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#leo solo los nombres de los archivos
path= 'leiblesDAT'
files = glob.glob(path + "/*.csv")
diccionarioDatos={}
for filename in files:
    diccionarioDatos[filename[(len(path)+1):len(filename)-4]]=pd.read_csv(filename,index_col=None,on_bad_lines='skip', delimiter="\t", header=None)


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


#Dataset custom para el encoder
datasetCustom =    pd.read_csv('oneHotTrainingResumido.csv',index_col=None,on_bad_lines='skip', sep='\s+',header=None)
datasetCustom.iloc[:, 0] = np.array([to_seconds(t) for t in datasetCustom.iloc[:, 0]])
#entrenamos el one hot encoder
labelencoder=LabelEncoder()
datasetCustom.iloc[:,2]=labelencoder.fit_transform(datasetCustom.iloc[:,2])
transformer = ColumnTransformer( transformers=[("ecg",OneHotEncoder(categories='auto'),[2])], remainder='passthrough')
datasetCustom=transformer.fit_transform(datasetCustom)




#vamos a construir el train test split
numeros_posibles=sorted(list(diccionarioDatos.keys()))#[2:3] por si quiero trabajar solo con un ecg
#si voy a probar solo con los ficheros de anotaciones N
# numeros_posibles=[100, 101, 103, 105, 108, 112, 113, 114, 115, 116, 117,
#            121, 122, 123, 201, 202, 205, 209, 215, 219,
#            220,230 ,234]

factor= int(len(numeros_posibles)*0.3) #proporcion train/test
# factor=1 #por si voy a trabajar solo con un ecg

#train-test split
x_test = random.sample(numeros_posibles, factor) #x numero de ecgs para test
x_train=[]
for numero in numeros_posibles:
    if numero not in x_test: #si no quiero overfitear xtest!=xtrain
    # if numero  in x_test: #si quiero overfitear xtest==xtrain
        x_train.append(numero)


#construimos todas las rutas según los numeros que tenemos para train y los que tenemos para test
path_x="./senales_troceadas/ecgs/"
path_y="./senales_troceadas/anotaciones_reducidas/"

lista_paths_test_y=[]
lista_paths_test_x=[]

for numero_test in x_test:
    filesx = glob.glob(path_x+str(numero_test)+"/*.csv")
    lista_paths_test_x.extend(filesx)
    filesy=glob.glob(path_y+str(numero_test)+"/*.csv")
    lista_paths_test_y.extend(filesy)

lista_paths_train_y=[]
lista_paths_train_x=[]
for numero_train in x_train:
    filesx2 = glob.glob(path_x+str(numero_train)+"/*.csv")
    lista_paths_train_x.extend(filesx2)
    filesy2 = glob.glob(path_y+str(numero_train)+"/*.csv")
    lista_paths_train_y.extend(filesy2)

#ordenamos las listas para mantener el orden en las particiones de los ecgs
lista_paths_test_y=sorted(lista_paths_test_y)
lista_paths_test_x=sorted(lista_paths_test_x)
lista_paths_train_y=sorted(lista_paths_train_y)
lista_paths_train_x=sorted(lista_paths_train_x)

#si quiero overfittear con el archivo 102 por ejemplo
# if lista_paths_train_x[0] == './senales_troceadas/ecgs/102/102_part001.csv':
#     lista_paths_train_x=lista_paths_test_x=lista_paths_train_x
#     lista_paths_train_y=lista_paths_test_y=lista_paths_train_y
# elif lista_paths_test_x[0] == './senales_troceadas/ecgs/102/102_part001.csv':
#     lista_paths_train_x=lista_paths_test_x=lista_paths_test_x
#     lista_paths_train_y=lista_paths_test_y=lista_paths_test_y
#

print(lista_paths_train_x)
print(lista_paths_train_y)
print(lista_paths_test_x)
print(lista_paths_test_y)

class CustomDataGenerator(Sequence):
    def __init__(self, x_filenames, y_filenames, batch_size):
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_filenames = self.x_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_filenames = self.y_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]


        batch_x =np.asarray([np.loadtxt(filename) for filename in batch_x_filenames]).astype(np.float32)

        batch_y = np.asarray([np.loadtxt(filename,delimiter=',') for filename in batch_y_filenames]).astype(np.float32)


        return batch_x, batch_y

train_data_generator = CustomDataGenerator(lista_paths_train_x, lista_paths_train_y, batch_size=1)

test_data_generator = CustomDataGenerator(lista_paths_test_x, lista_paths_test_y, batch_size=1)


##red neuronal cutre
# model = keras.models.Sequential([
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(6, activation="softmax")
# ])


# Transformer
input_shape = (5000, 2)
# Capas convolucionales de extracción de características
inputs = Input(shape=input_shape)
x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
# x = Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
# x = Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
# x = Conv1D(512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
# Capa del transformer

# sine_encoder = SinePositionEncoding(max_wavelength=5000)
# sine_encoding = sine_encoder(x)
# encoded=x+sine_encoding
#encoder
encoder_output=TransformerEncoder(num_heads=64, intermediate_dim=128, dropout=0.3)(encoded)
# Decoder layers
# decoder_input = SinePositionEncoding(max_wavelength=5000)(encoder_output)
# decoder_output = TransformerDecoder( num_heads=64, intermediate_dim=128, dropout=0.3)(encoder_output)

#classifier layers
# classif=UpSampling1D(8)(decoder_output)
classif = Conv1DTranspose(256, kernel_size=3, strides=8, padding="same", activation="relu")(encoder_output)
classif = Dense(256, activation="relu")(classif)
classif = Dropout(0.3)(classif)
classif = Dense(128, activation="relu")(classif)
classif = Dropout(0.3)(classif)
classif = Dense(6, activation="softmax")(classif)
model = tf.keras.Model(inputs=inputs, outputs=classif)

model.compile(optimizer="adam", loss="categorical_crossentropy")

model.fit_generator(train_data_generator, epochs=10)

test_predictions = model.predict(test_data_generator)

# # aquí mooving average NO AYUDA ASI QUE DE MOMENTO NO LA USO
# window=5
# test_predictions = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window), mode='same') / window, axis=0, arr=test_predictions)


print("ENTRENADO")
max_index = tf.argmax(test_predictions, axis=-1)
one_hot_output = tf.one_hot(max_index, depth=6)
one_hot_output = one_hot_output.numpy()


#
# plotear las salidas de la red neuronal con el ecg
for i in np.arange(len(lista_paths_test_x)):


    ecg_actual=np.asarray(np.loadtxt(lista_paths_test_x[i])).astype(np.float32)

    plt.plot(ecg_actual[:,0]+1)
    plt.title(str(i)+": "+lista_paths_test_x[i][29:-4])
    etiquetasactuales= np.asarray(np.loadtxt(lista_paths_test_y[i], delimiter=',')).astype(np.float32)
    # for k in np.arange(6):
    #
    #     plt.plot(np.arange(len(one_hot_output[i])),one_hot_output[i][:,k],label=str(k))
    #     plt.plot(etiquetasactuales[:,k], label="et: "+str(k),linestyle="dotted")
    #
    #     # plt.xlim(0,500)
    #     plt.ylim(-0,1.3)



    plt.plot(np.arange(len(one_hot_output[i])),one_hot_output[i][:,1],label=str(1),color="red")
    plt.plot(etiquetasactuales[:,1]-0.05, label="et: "+str(1),color="green")

    plt.plot(np.arange(len(one_hot_output[i])),one_hot_output[i][:,5]-0.2,label=str(5),color="orange")
    plt.plot(etiquetasactuales[:,5]-0.25, label="et: "+str(5),color="blue")
    # plt.xlim(0,500)
    plt.ylim(-0,1.3)

    plt.legend()
    plt.show()




decoded_labels = []
for i in range(one_hot_output.shape[0]): # iterar sobre las 14 filas
    for j in range(one_hot_output.shape[1]): # iterar sobre las 650000 muestras
        label_encoded = np.argmax(one_hot_output[i,j,:]) # obtener el índice del valor máximo
        label_decoded = labelencoder.inverse_transform([label_encoded])[0] # decodificar la etiqueta
        decoded_labels.append(label_decoded)

decoded_labels = np.transpose(np.asarray(decoded_labels).reshape(one_hot_output.shape[0], one_hot_output.shape[1]))

#Decoded_labels tiene forma de 5000 filas x 1820 columnas. Cada 130 columnas es un ecg nuevo de los 14 que son de test.
#Concatenamos las columnas de 130 en 130 de manera que tengamos 650000 filas x 14 columnas = los resultados para cada ecg
rg_min=0
rg_max=130
matriz_nueva = np.empty((650000, factor), dtype='U1')
for i in np.arange(factor):
    fila_actual=np.empty(0)
    contdor=0
    for k in np.arange(rg_min,rg_max):
        contdor+=1
        fila_actual=np.concatenate((fila_actual, decoded_labels[:, int(k)]))
    rg_min=rg_max
    rg_max=rg_max+130
    matriz_nueva[:,i]=fila_actual

#construimos la carpeta de resultados.
#para ello primero nos quedamos con los numeros de los archivos que eran de test
lista_archivos_resultado = set([elem[42:45] for elem in lista_paths_test_y])
fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
carpeta_actual = os.path.join("./resultados/", fecha_actual)
os.makedirs(carpeta_actual, exist_ok=True)



#construimos el fichero de anotacciones siguiendo la estructura de las anotaciones de los cardiologos
frecMuestreo=360
for col_num in np.arange(len(lista_archivos_resultado)):
    f = open ("./resultados/"+fecha_actual+"/"+str(sorted(list(lista_archivos_resultado))[col_num])+".csv",'w')
    columna_actual=matriz_nueva[:,col_num]
    matriz=np.column_stack((np.arange(650000),columna_actual))
    # deshacer la window, nos quedamos con el del medio
    for indice in np.arange(650000):
        if matriz[indice,1]=='Z':
            continue
        else:
            latido_actual=matriz[indice,1]
            indice_adelantado=indice
            while(True):
                if indice_adelantado>=650000:
                    break

                elif latido_actual == matriz[indice_adelantado,1]:
                    indice_adelantado+=1
                else:
                    break
            ancho_latido= indice_adelantado-indice
            for muestra in np.arange(indice, indice_adelantado):
                if muestra== int((indice+indice_adelantado)/2):
                    continue
                else:
                    matriz[muestra,1]='Z'
            indice=indice_adelantado
    # anotaciones
    for indice in np.arange(650000):
        if matriz[indice,1]=='Z':
            continue
        else:
            tiempo=int(matriz[indice,0])/360
            tiempostr=segundos_a_segundos_minutos_y_horas(tiempo)
            stringEspacios=cadenaEspacios(len(tiempostr))
            stringtiempo= stringEspacios+tiempostr
            string=stringtiempo+'{:9d}'.format(int(matriz[indice,0]))+'     '+matriz[indice,1]+'{:5d}{:5d}{:5d}'.format(0,0,0)+"\n"
            f.write(string)
    f.close()





ruta_directorio=os.getcwd()
ruta_carpeta = os.path.join(ruta_directorio, "resultados/"+fecha_actual)

for fichero_atr in sorted(lista_archivos_resultado):
    ruta_archivo = os.path.join(ruta_directorio, fichero_atr+".atr")
    ruta_copia = os.path.join(ruta_carpeta, fichero_atr+".atr")
    shutil.copy(ruta_archivo, ruta_copia)

for fichero_hea in sorted(lista_archivos_resultado):
    ruta_archivo = os.path.join(ruta_directorio, fichero_hea+".hea")
    ruta_copia = os.path.join(ruta_carpeta, fichero_hea+".hea")
    shutil.copy(ruta_archivo, ruta_copia)


carpeta_actual2 = os.path.join("./MyqrsLeible/", fecha_actual)
os.makedirs(carpeta_actual2, exist_ok=True)
for nombrefichero in sorted(lista_archivos_resultado):

    os.system("cat ./resultados/"+fecha_actual+"/"+nombrefichero+".csv | wrann -r ./resultados/"+fecha_actual+"/"+nombrefichero+" -a myqrs")

    os.system("rdann -r ./resultados/"+fecha_actual+"/"+nombrefichero+" -a myqrs>./MyqrsLeible/"+fecha_actual+"/"+nombrefichero+".csv")

    os.system("bxb -r ./resultados/"+fecha_actual+"/"+nombrefichero +" -a atr myqrs >> ./resultados/"+fecha_actual+"/resultados_bxb.txt")


print("acabbaod")
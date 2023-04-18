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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import random
from tensorflow import keras
from keras.utils import Sequence
import tensorflow as tf
from datetime import datetime
import shutil
from keras.layers import Input, Conv1D, MaxPooling1D, LayerNormalization, Flatten, Dense, Dropout, MultiHeadAttention, \
    GlobalMaxPooling1D, Reshape

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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

    for key in diccionarioDatos.keys():
        print(key)
        canal1=diccionarioDatos[key][1]
        canal1_normalizado = (canal1 - canal1.mean()) / canal1.std()
        diccionarioDatos[key][1]=canal1_normalizado

        canal2=diccionarioDatos[key][2]
        canal2_normalizado = (canal2 - canal2.mean()) / canal2.std()
        diccionarioDatos[key][2]=canal2_normalizado

        print(len(diccionarioDatos[key]))
        diccionarioDatos[key].iloc[:,1:3].to_csv("./datosNormalizados/"+key+".csv", index=False, header=False,sep=' ')


diccionarioAnotacionesCeros={}

def window_separado(bandwidth):
    #me dan la bandwith en seegundos pero la pasamos a muestras
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
        #la ultima fila no la pongo que es mala





# fillzeros()

# window()





# normalizarX()



#importamos el dataset para el encoder
datasetCustom =    pd.read_csv('oneHotTraining.csv',index_col=None,on_bad_lines='skip', sep='\s+',header=None)

datasetCustom.iloc[:, 0] = np.array([to_seconds(t) for t in datasetCustom.iloc[:, 0]])
datasetCustom.iloc[10, 2] ='""""'
#entrenamos el one hot encoder

labelencoder=LabelEncoder()
datasetCustom.iloc[:,2]=labelencoder.fit_transform(datasetCustom.iloc[:,2])

transformer = ColumnTransformer( transformers=[("ecg",OneHotEncoder(categories='auto'),[2])], remainder='passthrough')
datasetCustom=transformer.fit_transform(datasetCustom)
datasetCustom = datasetCustom.toarray()




#YA TENGO EL ENCODER APLICADO EN LOS DATOS PORQUE NO SE DONDE METERLO EN LO DE LOS BATCH!!!!


# diccionarioAnotacionesencoder={}

# path= 'con_window_separado'

# files = glob.glob(path + "/*.csv")

# for filename in files:
#     data = []
#     with open(filename) as file:
#         for line in file:
#             try:
#                 line_data = line.strip().split()
#                 data.append(line_data)
#             except:
#                 continue
#     diccionarioAnotacionesencoder[filename[(len(path)+1):len(filename)-4]] = pd.DataFrame(data)
#     fic=filename[(len(path)+1):len(filename)-4]

#     print(fic)




#     diccionarioAnotacionesencoder[fic][2]=labelencoder.transform(diccionarioAnotacionesencoder[fic][2])




#     aux=transformer.transform(diccionarioAnotacionesencoder[fic]).toarray()
#     a=diccionarioAnotacionesencoder[fic]

#     columns_to_keep = aux[:, [24, 25]]
#     aux = np.concatenate((columns_to_keep, aux[:, :24], aux[:, 26:]), axis=1)

#     newpandas=pd.DataFrame(aux)
#     newpandas.iloc[:,2:-3].to_csv("./window_encoder/"+fic+".csv", index=False, header=False)

#     diccionarioAnotacionesencoder.pop(filename[(len(path)+1):len(filename)-4])






numeros_posibles=list(diccionarioDatos.keys())



#voy a probar solo con los ficheros de anotaciones N
# numeros_posibles=[100, 101, 103, 105, 108, 112, 113, 114, 115, 116, 117,
#            121, 122, 123, 201, 202, 205, 209, 215, 219,
#            220,230 ,234]
factor= int(len(numeros_posibles)*0.3)

x_test = random.sample(numeros_posibles, factor)


x_train=[]
for numero in numeros_posibles:
    if numero not in x_test:
        x_train.append(numero)

#troceo de 5000
path_x="./senales_troceadas/ecgs/"
path_y="./senales_troceadas/anotaciones/"


# #troceo de 10000
# path_x="./senales_troceada2/ecgs/"
# path_y="./senales_troceada2/anotaciones/"

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

lista_paths_test_y=sorted(lista_paths_test_y)
lista_paths_test_x=sorted(lista_paths_test_x)
lista_paths_train_y=sorted(lista_paths_train_y)
lista_paths_train_x=sorted(lista_paths_train_x)

class CustomDataGenerator(Sequence):
    # def __init__(self, x_filenames, y_filenames, batch_size, block_size, l=650000):
    def __init__(self, x_filenames, y_filenames, batch_size):
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size
        # self.l = l

    def __len__(self):
        return int(np.ceil(len(self.x_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_filenames = self.x_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_filenames = self.y_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]


        batch_x =np.asarray([np.loadtxt(filename) for filename in batch_x_filenames]).astype(np.float32)
        batch_y = np.asarray([np.loadtxt(filename,delimiter=',') for filename in batch_y_filenames]).astype(np.float32)


        return batch_x, batch_y

train_data_generator = CustomDataGenerator(lista_paths_train_x, lista_paths_train_y, batch_size=130)

test_data_generator = CustomDataGenerator(lista_paths_test_x, lista_paths_test_y, batch_size=130)

# model = keras.models.Sequential([
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(24, activation="softmax")
# ])

# Especificar la forma de entrada
input_shape = (5000, 2)
# Capas convolucionales de extracción de características
inputs = Input(shape=input_shape)

x = Conv1D(8, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)

x = Conv1D(8, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(8, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = Conv1D(8, kernel_size=3, strides=1, padding="same", activation="relu")(x)
# x = GlobalMaxPooling1D()(x)
# x = Reshape((-1, 64))(x)
x = MaxPooling1D(pool_size=2)(x)
# Capa del transformer
query = LayerNormalization()(x)
key = LayerNormalization()(x)
value = LayerNormalization()(x)
attention_output = MultiHeadAttention(num_heads=2, key_dim=8, dropout=0.3)(query, value, key)
x = LayerNormalization()(x + attention_output)
# Capa de clasificación
x = Flatten()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.3)(x)

x = Dense(24, activation="softmax")(x)
# Crear modelo
print(x)
model = tf.keras.Model(inputs=inputs, outputs=x)
model.compile(optimizer="adam", loss="categorical_crossentropy")
print(train_data_generator)
model.fit_generator(train_data_generator, epochs=10)

test_predictions = model.predict(test_data_generator)
#aquí mooving average

# for i in np.arange(len(lista_paths_test_x)):
#
#
#     ecg_actual=np.asarray(np.loadtxt(lista_paths_test_x[i])).astype(np.float32)
#
#     plt.plot(ecg_actual[:,0])
#     plt.title(lista_paths_test_x[i][29:-4])
#     for k in np.arange(24):
#         plt.plot(np.arange(len(test_predictions[i])),test_predictions[i][:,k],label=str(k))
#         # plt.xlim(0,1000)
#         # plt.ylim(-0.1,1.1)
#     plt.legend()
#     plt.figure()
#
#     if i==300:
#         break

max_index = tf.argmax(test_predictions, axis=-1)

# window=5
# test_predictions_window = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window), mode='same') / window, axis=0, arr=test_predictions)
# max_index = tf.argmax(test_predictions_window, axis=-1)

# max_index = tf.argmax(test_predictions, axis=-1)

one_hot_output = tf.one_hot(max_index, depth=24)

one_hot_output = one_hot_output.numpy()





decoded_labels = []
for i in range(one_hot_output.shape[0]): # iterar sobre las 14 filas
    for j in range(one_hot_output.shape[1]): # iterar sobre las 650000 muestras
        label_encoded = np.argmax(one_hot_output[i,j,:]) # obtener el índice del valor máximo
        label_decoded = labelencoder.inverse_transform([label_encoded])[0] # decodificar la etiqueta
        decoded_labels.append(label_decoded)

decoded_labels = np.transpose(np.asarray(decoded_labels).reshape(one_hot_output.shape[0], one_hot_output.shape[1]))


matriz_nueva = np.empty((650000, factor),dtype='U1')
rg_min=0
rg_max=130#para separacion de 5000
# rg_max=65#para separacion de 10000
for i in np.arange(factor):
    fila_actual=np.empty(0)
    contdor=0
    for k in np.arange(rg_min,rg_max):
        contdor+=1
        fila_actual=np.concatenate((fila_actual, decoded_labels[:, int(k)]))
    rg_min=rg_max
    rg_max=rg_max+130#para separacion de 5000
    # rg_max=rg_max+65#para separacion de 10000
    matriz_nueva[:,i]=fila_actual

lista_archivos_resultado = set([elem[32:35] for elem in lista_paths_test_y])


fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


carpeta_actual = os.path.join("./resultados/", fecha_actual)
os.makedirs(carpeta_actual, exist_ok=True)




frecMuestreo=360
for col_num in np.arange(len(lista_archivos_resultado)):
    f = open ("./resultados/"+fecha_actual+"/"+str(sorted(list(lista_archivos_resultado))[col_num])+".csv",'w')

    columna_actual=matriz_nueva[:,col_num]
    matriz=np.column_stack((np.arange(650000),columna_actual))


    # deshacer la window
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




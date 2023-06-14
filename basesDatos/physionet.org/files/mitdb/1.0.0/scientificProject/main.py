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
import scipy.signal as sig
from keras_nlp.layers import SinePositionEncoding, TransformerDecoder,TransformerEncoder
from sklearn.model_selection import train_test_split
from receptivefield.keras import KerasReceptiveField

from keras.layers import Input, Conv1D, MaxPooling1D, LayerNormalization, Flatten, Dense, Dropout, MultiHeadAttention, \
    GlobalMaxPooling1D, Reshape, UpSampling1D, Conv1DTranspose,BatchNormalization
from keras import regularizers

import randomTransformations

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
# numeros_posibles=[100, 101, 103, 105, 108, 112]



anotaciones_ficheros={'100': 'S', '101': 'S', '102': 'Q', '103': 'S', '104': 'Q', '105': 'Q', '106': 'V', '107': 'Q', '108': 'F', '109': 'F', '111': 'V', '112': 'S', '113': 'S', '114': 'F', '115': 'N', '116': 'S', '117': 'S', '118': 'S', '119': 'V', '121': 'S', '122': 'N', '123': 'V', '124': 'F', '200': 'F', '201': 'F', '202': 'F', '203': 'F', '205': 'F', '207': 'S', '208': 'F', '209': 'S', '210': 'F', '212': 'N', '213': 'F', '214': 'F', '215': 'F', '217': 'Q', '219': 'F', '220': 'S', '221': 'V', '222': 'S', '223': 'F', '228': 'S', '230': 'V', '231': 'S', '232': 'S', '233': 'F', '234': 'S'}
label_per_file= list(anotaciones_ficheros.values())
factor= int(len(numeros_posibles)*0.3) #proporcion train/test
# factor=1 #por si voy a trabajar solo con un ecg

# #train-test split
# x_test = random.sample(numeros_posibles, factor) #x numero de ecgs para test
# print(x_test)
# x_train=[]
# for numero in numeros_posibles:
#     if numero not in x_test: #si no quiero overfitear xtest!=xtrain
#     # if numero  in x_test: #si quiero overfitear xtest==xtrain
#         x_train.append(numero)


x_train, x_test = train_test_split(numeros_posibles,test_size=factor,stratify= label_per_file)
x_train=sorted(x_train)
x_test=sorted(x_test)


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
#
# print(lista_paths_train_x)
# print(lista_paths_train_y)
# print(lista_paths_test_x)
# print(lista_paths_test_y)




class CustomDataGenerator(Sequence):
    def __init__(self, x_filenames, y_filenames, batch_size, train):
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size
        self.train = train

    def __len__(self):
        return int(np.ceil(len(self.x_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_filenames = self.x_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_filenames = self.y_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]


        batch_x =np.asarray([np.loadtxt(filename) for filename in batch_x_filenames]).astype(np.float32)

        #trasnformaciones al ecg?
        #batch_x=transformacion(batch_x)

        batch_y = np.asarray([np.loadtxt(filename,delimiter=',') for filename in batch_y_filenames]).astype(np.int32)
        if self.train==True:
            random_number=np.random.randint(0,7)#devuelve hasta el 5
            # random_number=0

            # print(random_number)
            # plt.plot(batch_x[0,:,0]+2,label="primero sin filt")
            # plt.plot(batch_x[0,:,1]+1,label="segundo sin filt")
            batch_x, batch_y=randomTransformations.randomTransformation(random_number,batch_x,batch_y,titulo=True)
            # plt.plot(batch_x[0,:,0],label="primero filtrado")
            # plt.plot(batch_x[0,:,1],label="seugndo siltrado")
            # plt.xlim(0,500)
            # plt.legend()
            # plt.show()
            # plt.figure()

        return batch_x, batch_y

# lista_paths_train_x = lista_paths_train_x[:260]
# lista_paths_train_y = lista_paths_train_y[:260]
# lista_paths_test_x = lista_paths_test_x[:260]
# lista_paths_test_y = lista_paths_test_y[:260]

train = CustomDataGenerator(lista_paths_train_x, lista_paths_train_y, batch_size=1,train=True)
test = CustomDataGenerator(lista_paths_test_x, lista_paths_test_y, batch_size=1,train=False)
##red neuronal cutre
# model = keras.models.Sequential([
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(6, activation="softmax")
# ])






input_shape = (5000, 2)
# Capas convolucionales de extracción de características
def model_build_func(input_shape,kernel_conv=7,strides_conv=1,dilation_rate=2):
    #antes tenia kernel 3 y strides 1
    inputs = Input(shape=input_shape,name='input_lay')
    x = Conv1D(64, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate,padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = Conv1D(64, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate,padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x=LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate,padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Conv1D(64, kernel_size=kernel_conv, strides=strides_conv,dilation_rate=dilation_rate, padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x=LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate,padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Conv1D(128, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate, padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x=LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate,padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Conv1D(128, kernel_size=kernel_conv, strides=strides_conv,dilation_rate=dilation_rate, padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x=LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(512, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate,padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Conv1D(512, kernel_size=kernel_conv, strides=strides_conv,dilation_rate=dilation_rate, padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x=LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(512, kernel_size=kernel_conv, strides=strides_conv,dilation_rate=dilation_rate, padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Conv1D(512, kernel_size=kernel_conv, strides=strides_conv, dilation_rate=dilation_rate,padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001), name='last_conv')(x)
    x=LayerNormalization(-2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)
    # Capa del transformer
    # sine_encoder = SinePositionEncoding(max_wavelength=x.shape[1])
    # sine_encoding = sine_encoder(x)
    # encoded=x+sine_encoding
    # encoded=LayerNormalization(-2)(encoded)
    # encoder_output=TransformerEncoder(num_heads=128, intermediate_dim=128, dropout=0.3)(encoded)
    #
    # # Decoder layers
    # decoder_input = encoder_output
    # decoder_output = keras_nlp.layers.TransformerDecoder( num_heads=128, intermediate_dim=128, dropout=0.3)(decoder_input)

    # classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(x)
    # classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(classif)
    # classif = Conv1DTranspose(256, kernel_size=8, strides=2, padding="same", activation="elu",kernel_regularizer=regularizers.l2(0.0001))(classif)
    classif=UpSampling1D(8)(x)

    classif = Dense(256, activation="elu")(classif)
    classif=LayerNormalization(-2)(classif)
    classif = BatchNormalization()(classif)
    classif = Dropout(0.3)(classif)


    classif = Dense(128, activation="elu")(classif)
    classif=LayerNormalization(-2)(classif)

    classif = BatchNormalization()(classif)
    classif = Dropout(0.3)(classif)
    classif = Dense(6, activation="softmax")(classif)
    model = tf.keras.Model(inputs=inputs, outputs=classif)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model






# wrapper que transforma el keras sequence en un generator
def train_data_generator():
    for i in range(len(train)):
        x, y = train[i]
        yield x[0], y[0]   # x[0] e y[0] para quitarle la dimensión de batch (ver luego dataset.batach)

# ... para ahora transformarlo a un Dataset
datasettrain = tf.data.Dataset.from_generator(train_data_generator, output_signature=(tf.TensorSpec(shape=(5000, 2), dtype=tf.float32),
                                                                                 tf.TensorSpec(shape=(5000, 6), dtype=tf.int32)))




def test_data_generator():
    for i in range(len(test)):
        x, y = test[i]
        yield x[0], y[0]   # x[0] e y[0] para quitarle la dimensión de batch (ver luego dataset.batach)

# ... para ahora transformarlo a un Dataset
datasettest = tf.data.Dataset.from_generator(test_data_generator, output_signature=(tf.TensorSpec(shape=(5000, 2), dtype=tf.float32),
                                                                                 tf.TensorSpec(shape=(5000, 6), dtype=tf.int32)))







model=model_build_func(input_shape)

datasettrain = datasettrain.shuffle(len(train)).batch(1)  # si len(train) es demasiado grande, reducir el tamano del buffer
datasettest = datasettest.batch(1)  # si len(train) es demasiado grande, reducir el tamano del buffer



history =model.fit_generator(datasettrain, epochs=300,validation_data=datasettest)
train_loss = history.history['loss']
test_loss = history.history['val_loss']
#

def define_ma(L):
    return np.repeat(1. / (2 * L + 1), 2 * L + 1)
test_predictions = model.predict(datasettest)


#para el ancho de la mooving average le paso un segundo que es lo que mide un latido a ver como sale
L=61
b=define_ma(L)

for i in np.arange(len(test_predictions)):
    for k in np.arange(test_predictions[i].shape[1]):
        test_predictions[i][:,k]=sig.filtfilt(b,1,test_predictions[i][:,k])

print("ENTRENADO")
max_index = tf.argmax(test_predictions, axis=-1)
one_hot_output = tf.one_hot(max_index, depth=6)
one_hot_output = one_hot_output.numpy()


# plotear las salidas de la red neuronal con el ecg
#
# for i in np.arange(len(lista_paths_test_x)):
#
#
#
#
#     ecg_actual=np.asarray(np.loadtxt(lista_paths_test_x[i])).astype(np.float32)
#     decoder_string="F = 0\nN = 1\nQ = 2\nS = 3\nV = 4\nZ = 5"
#     plt.plot(ecg_actual[:,0]+1,label=decoder_string)
#     plt.title(str(i)+": "+lista_paths_test_x[i][29:-4])
#     etiquetasactuales= np.asarray(np.loadtxt(lista_paths_test_y[i], delimiter=',')).astype(np.float32)
#     for k in np.arange(5):
#
#
#
#         plt.plot(np.arange(len(one_hot_output[i])),one_hot_output[i][:,k],label="PRED: "+str(k))
#         plt.plot(etiquetasactuales[:,k]-0.05-(k/100), label="ORIG: "+str(k))
#
#     plt.plot(np.arange(len(one_hot_output[i])),one_hot_output[i][:,5]-0.4,label="PRED: "+str(5),color="brown")
#     plt.plot(etiquetasactuales[:,5]-0.45, label="ORIG: "+str(5),color="black")
#
#
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.subplots_adjust(right=0.75)
#
#     os.makedirs("plots", exist_ok=True)
#
#     # Guardar la figura en la carpeta con un nombre único
#     # plt.savefig("plots/" +str(i)+": "+lista_paths_test_x[i][29:-4]+ ".png")
#     #
#     plt.show()
#     plt.figure()
#




decoded_labels = []
for i in range(one_hot_output.shape[0]): # iterar sobre las 14 filas
    print(str(i)+"/"+str(one_hot_output.shape[0]))
    for j in range(one_hot_output.shape[1]): # iterar sobre las 650000 muestras
        label_encoded = np.argmax(one_hot_output[i,j,:]) # obtener el índice del valor máximo
        label_decoded = labelencoder.inverse_transform([label_encoded])[0] # decodificar la etiqueta
        decoded_labels.append(label_decoded)

decoded_labels = np.transpose(np.asarray(decoded_labels).reshape(one_hot_output.shape[0], one_hot_output.shape[1]))
print("decoded labels hecho")
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

for fichero_qrs in sorted(lista_archivos_resultado):
    ruta_archivo = os.path.join(ruta_directorio, fichero_qrs+".qrs")
    ruta_copia = os.path.join(ruta_carpeta, fichero_qrs+".qrs")
    shutil.copy(ruta_archivo, ruta_copia)


carpeta_actual2 = os.path.join("./MyqrsLeible/", fecha_actual)
os.makedirs(carpeta_actual2, exist_ok=True)
for nombrefichero in sorted(lista_archivos_resultado):

    os.system("cat ./resultados/"+fecha_actual+"/"+nombrefichero+".csv | wrann -r ./resultados/"+fecha_actual+"/"+nombrefichero+" -a myqrs")

    os.system("rdann -r ./resultados/"+fecha_actual+"/"+nombrefichero+" -a myqrs>./MyqrsLeible/"+fecha_actual+"/"+nombrefichero+".csv")
    os.system("echo MY_ANNOTATIONS: >> ./resultados/"+fecha_actual+"/resultados_bxb.txt")
    os.system("bxb -r ./resultados/"+fecha_actual+"/"+nombrefichero+" -a atr myqrs >> ./resultados/"+fecha_actual+"/resultados_bxb.txt")
    os.system("echo GQRS_ANNOTATIONS >> ./resultados/"+fecha_actual+"/resultados_bxb.txt")
    os.system("bxb -r ./resultados/"+fecha_actual+"/"+nombrefichero+" -a atr qrs >> ./resultados/"+fecha_actual+"/resultados_bxb.txt")
    os.system("echo ---------------------------------------------------------------------------------------------------------------- >> ./resultados/"+fecha_actual+"/resultados_bxb.txt")

plt.figure()
print(train_loss)
print(test_loss)
plt.plot(train_loss,label="train_loss")
plt.plot(test_loss,label="test_loss")
plt.legend()
plt.savefig("./resultados/"+fecha_actual+"/loss.png")
print("acabbaod")
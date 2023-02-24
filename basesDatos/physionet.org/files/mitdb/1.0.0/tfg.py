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
from tensorflow.keras.utils import Sequence

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
#     newpandas.to_csv("./window_encoder/"+fic+".csv", index=False, header=False)

#     diccionarioAnotacionesencoder.pop(filename[(len(path)+1):len(filename)-4])





factor= int(48*0.3)#el 30% de los archivos va a ser para predecir

numeros_posibles=list(diccionarioDatos.keys())
x_test = random.sample(numeros_posibles, factor)

x_train=[]
for numero in numeros_posibles:
    if numero not in x_test:
        x_train.append(numero)
    
path_x="./datosNormalizados/"

path_y="./window_encoder/"

lista_paths_test_y=[]
lista_paths_test_x=[]

for numero_test in x_test:
    lista_paths_test_x.append(path_x+str(numero_test)+".csv")

    lista_paths_test_y.append(path_y+str(numero_test)+".csv")

lista_paths_train_y=[]
lista_paths_train_x=[]
for numero_train in x_train:
    lista_paths_train_x.append(path_x+str(numero_train)+".csv")
    lista_paths_train_y.append(path_y+str(numero_train)+".csv")


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
        batch_y = np.asarray([np.loadtxt(filename,delimiter=",") for filename in batch_y_filenames])[0].astype(np.float32)
       

        return batch_x, batch_y


train_data_generator = CustomDataGenerator(lista_paths_train_x, lista_paths_train_y, batch_size=8)
test_data_generator = CustomDataGenerator(lista_paths_test_x, lista_paths_test_y, batch_size=8)


model = keras.models.Sequential([
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(29)
])

model.compile(optimizer="adam", loss="mse")

model.fit_generator(train_data_generator, epochs=5)

test_predictions = model.predict(test_data_generator)

# test_loss = model.evaluate(test_data_generator, test_predictions)

#matrix del mas alto
#desencodificar

# # f = open (nombrefichero+".csv",'w')

# # frecMuestreo=360
# # for pico in picos:
# #     tiempo=pico/360
# #     tiempostr=segundos_a_segundos_minutos_y_horas(tiempo)
    
    
# #     stringEspacios=cadenaEspacios(len(tiempostr))
# #     stringtiempo= stringEspacios+tiempostr
    
# #     string=stringtiempo+'{:9d}'.format(pico)+'     N{:5d}{:5d}{:5d}'.format(0,0,95)+"\n"
    



            
            
# #     f.write(string)



    
#     # os.system("cat "+nombrefichero+".csv | wrann -r "+nombrefichero+" -a myqrs")
    
#     # os.system("rdann -r "+nombrefichero+" -a myqrs>./MyqrsLeible/"+nombrefichero+".csv") 

#     # os.system("bxb -r "+nombrefichero +" -a atr myqrs")




# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:32:48 2022

@author: Dania
"""
#ejercicio de averiguar que clientes dejan el banco

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Churn_Modelling.csv')

x=dataset.iloc[:,3:13].values #aqui ya elegimos qué columnas nos son interesantes para este estudio, por ejemplo el apellido nos da igual
#queremos de la tres a la 12
y=dataset.iloc[:,13].values

#pasar variables categoricas a variables dummy
#las variables dummy se codifican por separado para que las columnas codificadas no compartan codigo

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#codificamos a numeros 1,2,3 
labelEnconder_XPais= LabelEncoder()
x[:,1]= labelEnconder_XPais.fit_transform(x[:,1])
labelEnconder_XSex= LabelEncoder()
x[:,2]= labelEnconder_XSex.fit_transform(x[:,2])

#para no tener en cuenta el orden y tal, como son variables categoricas pasamos a 1 y 0
from sklearn.compose import ColumnTransformer
#el onehotencoder esta obsoleto, ahora hay que usar el column transformer creando un transformer llamado churn_modeling,//  OneHotEncoder(categories='auto'), == La clase a la que transformar// [1]# Las columnas a transformar.
transformer = ColumnTransformer( transformers=[("Churn_Modelling",OneHotEncoder(categories='auto'),[1])], remainder='passthrough')
#aplicamos el transformer
x = transformer.fit_transform(x)
x = x[:, 1:]



#dividimos entre conjunto de prueba y de entrenamiento
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state = 0)



#normalizamos las variables
from sklearn.preprocessing import StandardScaler 
sc_X= StandardScaler()
xtrain=sc_X.fit_transform(xtrain)
xtest=sc_X.transform(xtest) #solo hago transform y no fit transform para que se aplique la misma transformacion que se ha aplicado arriba


#importar las librerías de keras
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout

#inicializar la red neuronal

classifier= Sequential()

#añadir las capas de entrada y primera capa oculta de la red neuronal
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11 ))#elegimos 6 porque en la capa oculta porque es la media entre los nodos de la capa de entrada y la capa de salida
#queremos utilizar el rectificador lineal unitario como funcion de activacion y el inicializador de los pesos lo hacemos con la funcion uniforme
#input dim es el numero de nodos de la capa actual

#metemos una segunda capa oculta
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu" ))
#aqui iria inputdim=6 que es la dimension de la capa anterior pero no hace falta ponerlo porque esta nueva capa ya sabe que se tiene que conectar con la anterior

#metemos la capa de salida
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid" ))
#esta solo tiene una slaida, 0 o 1 si se queda en el bancoo se va// en la capa de salida me interesa tener una prob, asi que la funcion de activacion va a ser la sigmoide



#compilar la red neuronal
classifier.compile(optimizer="adam", loss= "binary_crossentropy", metrics=["accuracy"])
#el metodo de optimizacion es el de adam
#la funcion de perdidas usamos la binary crossentropy



#entrenar el modelo 
classifier.fit(xtrain, ytrain, batch_size=10, epochs= 100)


#predecimos a los nuevos

ypred= classifier.predict(xtest)

#elegimos un umbral, vamos a determinar que los potenciales clientes en irse son los que tengan mas de 50%

ypred= (ypred>0.5)

#crear una matriz de confusion  para verlo mejor
#se ha hecho una buena prediccion porque hay muchos clientes en la diagonal de la matriz, la suma de las diagonales entre el total nos da la fiabilidad
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest,ypred)

fiabilidad= (cm[0,0]+cm[1,1])/np.sum(cm)



#evaluar  la red neuronal


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
 # classifier.add(Dropout(p = 0.1))

  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
 # classifier.add(Dropout(p = 0.1))

  classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

  classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator=classifier, X = xtrain, y = ytrain, cv = 10, n_jobs=-1, verbose = 1)
mean= accuracies.mean()
variance=accuracies.std()




#mejorar la red neuronal mejorando los hiperparametros

from sklearn.model_selection import GridSearchCV
#dicccionario para los parametros que queremos ajustar, va a construir la mejor red neuronal con todas las combinaciones de los parametros que especificamos en el diccionario
def build_classifier(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
 # classifier.add(Dropout(p = 0.1))

  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
 # classifier.add(Dropout(p = 0.1))

  classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

  classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier


classifier = KerasClassifier(build_fn = build_classifier)

parameters={
    'batch_size': [25,32],
    'epochs':[100,500], 
    'optimizer':['adam','rmsprop']
    }

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring = 'accuracy',cv=10)


grid_search= grid_search.fit(xtrain,ytrain)

bestparameters=grid_search.best_params_
bestaccuracy=grid_search.best_score_

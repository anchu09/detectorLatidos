# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:05:22 2021

@author: Usuario
"""

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

x1= np.loadtxt("prueba1_features.txt")
x2= np.loadtxt("laura_features.txt")
x3= np.loadtxt("papa_features.txt")
x4= np.loadtxt("alex_features.txt")
x5= np.loadtxt("dani_features.txt")

lista_Datos= [x1,x2,x3,x4,x5]
lista_puntuacionesfinales_gmm=[]
for a in range(2):
    def reescalamiento(datos):
        return preprocessing.minmax_scale(datos)
    
    def log(datos):
        return np.log(0.0000001+datos)
    
    def transformacion_de_raiz(datos):
         return np.sqrt(datos+0.01)
    def Transformacion_Reciproca(datos):
         return 1/(datos+0.01)
    
    
    
    def choose_algorithm(clustering_type):
        if(clustering_type=="GMix"):  
                algoritmo= GaussianMixture(n_components=2, covariance_type="full", n_init=100)
        if(clustering_type=="KM"):  
                algoritmo= KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=100)
        if(clustering_type=="AJer"): 
                algoritmo= AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
        return algoritmo     
    
    def puntuacion(x, gaus, red_dim,clustering_type):
    
        x = reescalamiento(x)
    
            
        if (gaus=="L"):
            x = log(x)  
        elif (gaus=="SQRT"):
            x = transformacion_de_raiz(x) 
        elif (gaus=="Rcip"):
            x = Transformacion_Reciproca(x) 
            
      
        if (red_dim=="S_RDim"):
            
            pca= PCA(n_components=a+1)
        
            X_train, X_test = train_test_split(x,test_size=0.3)
               
            algoritmo = choose_algorithm(clustering_type)
           
            X_train = pca.fit_transform(X_train)
            
            algoritmo.fit(X_train)
            
            X_test = pca.transform(X_test)
        
        else:
        
            X_train, X_test = train_test_split(x,test_size=0.3)
            algoritmo = choose_algorithm(clustering_type)
            
            algoritmo.fit(X_train)
        
        if(clustering_type=="AJer"):         #como este algortimo no puede clasificar a diferencia de los demas, tenemos que usar un clasificador
            agrupameinto_jerarquico=algoritmo
            algoritmo=KNeighborsClassifier(n_neighbors=2, metric= 'minkowski', p=2)
            algoritmo.fit(X_train, agrupameinto_jerarquico.labels_)
    
      
        predicciones=algoritmo.predict(X_test)
    
        colores = ['red','blue']
        
        colores_cluster = [colores[predicciones[i]] for i in range(len(X_test))]
         
        plt.show()  
            
        db_index = davies_bouldin_score(X_test, predicciones)
       
       # plt.scatter(np.arange(X_test.shape[0]), X_test[:, 0],c=colores_cluster)
        
      #  plt.show()
    
        return db_index
    
    
    #para todas las muestras:
    lista_Resultados=[]
    lista_claves=[]
    
    lista_Gaussianizar= {"N_G","L","SQRT","Rcip"}
    lista_red_dim= {"N_RDim","S_RDim"}
    lista_clustering_type= {"GMix","AJer","KM"}
    datos_Comparacion_muestra={}
    datos_Comparacion_total=[]
    datos_comparacion_promediada={}
    contador=0
    for x in lista_Datos:
        contador=contador+1
        datos_Comparacion_muestra.clear()
        for gauss in lista_Gaussianizar:
            for red_dim in lista_red_dim:
                for clustering_type in lista_clustering_type:
                    clave= gauss+"-"+red_dim+"-"+clustering_type 
                    if (contador==1):
                        lista_claves.append(clave)
                    datos_Comparacion_muestra[clave]=puntuacion(x, gauss, red_dim, clustering_type)
                    lista_Resultados.append(puntuacion(x, gauss, red_dim, clustering_type))
        datos_Comparacion_total.append(datos_Comparacion_muestra.copy())
    
    promedio={}
    
    for tipo_de_puntuacion in range(len(lista_claves)):
        suma=0
        for sumatorio in range(len(lista_Datos)):
            suma+= lista_Resultados[tipo_de_puntuacion+len(lista_claves)*sumatorio]
               
        promedio[lista_claves[tipo_de_puntuacion]]=suma/len(lista_Datos)
    
    plot=plt.subplot()
    plt.bar(promedio.keys(), promedio.values(),width =0.5)
    plt.setp(plot.get_xticklabels(), rotation=90, ha='right')
    plt.title('Promedio de cada combinación')
    plt.ylabel('Davies Bouldin Score')
    plt.show()
    lista_puntuacionesfinales_gmm.append(promedio.get("L-S_RDim-GMix"))
plt.bar(lista_puntuacionesfinales_gmm,range(10))





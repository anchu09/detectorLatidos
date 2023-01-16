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
# for num in np.arange(235):
#      os.system("rdsamp -r "+str(num)+">./leiblesDAT/"+str(num)+".csv")


#leemos todos los archivos de la carpeta

#
path= 'leiblesDAT'

files = glob.glob(path + "/*.csv")


diccionario={}

for filename in files:
    diccionario[filename[(len(path)+1):len(filename)-4]]=pd.read_csv(filename,index_col=None,on_bad_lines='skip', delimiter="\t")

#atr	reference beat, rhythm, and signal quality annotations



#leemos anotaciones


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



diccionarioPuntuaciones={}

def detector_latidos(datasetDATOS,nombrefichero):
    eje_x=datasetDATOS.iloc[:,-3].values
    latidos=datasetDATOS.iloc[:,-2].values
    picos,_=find_peaks(latidos, height=1100)
    plt.plot(picos, latidos[picos],"x",color="r")

    plt.plot(eje_x,latidos,color="b")
    plt.title(str(nombrefichero))
    plt.figure()
    
    

    f = open (nombrefichero+".csv",'w')

    frecMuestreo=360
    for pico in picos:
        tiempo=pico/360
        tiempostr=segundos_a_segundos_minutos_y_horas(tiempo)
        if (pico>0   and pico<9):
            string="\t"+tiempostr+"\t"+str(pico)+"   \tN\t0\t0\t95\n"

        elif (pico>9   and pico<100):
            string="\t"+tiempostr+"\t"+str(pico)+"  \tN\t0\t0\t95\n"

        
        elif (pico>99   and pico<1000):
            
            string="\t"+tiempostr+"\t"+str(pico)+" \tN\t0\t0\t95\n"
            
        else:string="\t"+tiempostr+"\t"+str(pico)+"\tN\t0\t0\t95\n"


                
                
        f.write(string)

    f.close()   


    #os.system("wrann -r "+nombrefichero+" -a myqrs")
    
    os.system("cat "+nombrefichero+".csv | wrann -r"+nombrefichero+" -a myqrs")
    
    os.system("rdann -r "+nombrefichero+" -a myqrs>./MyqrsLeible/"+nombrefichero+".csv") 

    # os.system("bxb -r "+nombrefichero +" -a atr myqrs")
    

    


    

for fichero in diccionario.keys():
    detector_latidos(diccionario[fichero],fichero)




















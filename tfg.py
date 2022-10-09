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
path= 'basesDatos/physionet.org/files/mitdb/1.0.0/leiblesDAT'

files = glob.glob(path + "/*.csv")


diccionario={}

for filename in files:
    diccionario[filename[(len(path)+1):len(filename)-4]]=pd.read_csv(filename,index_col=None,on_bad_lines='skip', delimiter="\t")

#atr	reference beat, rhythm, and signal quality annotations



#leemos anotaciones






diccionarioPuntuaciones={}

def detector_latidos(datasetDATOS,nombrefichero):
    eje_x=datasetDATOS.iloc[:,-3].values
    latidos=datasetDATOS.iloc[:,-2].values
    picos,_=find_peaks(latidos, height=1100)
    plt.plot(picos, latidos[picos],"x",color="r")

    plt.plot(eje_x,latidos,color="b")
    plt.title(str(nombrefichero))
    plt.figure()
    
    
    f = open ("basesDatos/physionet.org/files/mitdb/1.0.0/misAnotaciones/"+nombrefichero+".csv",'w')

    for pico in picos:
        
        if (pico>0   and pico<9):
            string="\t"+"0:00.000\t"+str(pico)+"   \tN\t0\t0\t0\n"

        elif (pico>9   and pico<100):
            string="\t"+"0:00.000\t"+str(pico)+"  \tN\t0\t0\t0\n"

        
        elif (pico>99   and pico<1000):
            
            string="\t"+"0:00.000\t"+str(pico)+" \tN\t0\t0\t0\n"
            
        else:string="\t"+"0:00.000\t"+str(pico)+"\tN\t0\t0\t0\n"


                
                
        f.write(string)

    f.close()   
    # f = open ("basesDatos/physionet.org/files/mitdb/1.0.0/misAnotaciones/"+nombrefichero+".csv",'w')

    os.system("wrann -r basesDatos/physionet.org/files/mitdb/1.0.0/misAnotaciones/"+nombrefichero+" -a qrs")
    

    


    

for fichero in diccionario.keys():
    detector_latidos(diccionario[fichero],fichero)




















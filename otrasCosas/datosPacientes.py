# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:06:20 2022

@author: Dania
"""

# importing the required modules
import glob
import pandas as pd
import numpy as np
  
# specifying the path to csv files
path= "C:/Users/Dania/Escritorio/carpetasleerjunto"  
# csv files in the path
files = glob.glob(path + "/*.csv")
  
# defining an empty list to store 
# content
data_frame = pd.DataFrame()
content = []
  

#creamos un diccionario para almacenar dataset para cada nombre de archivo
diccionario={}

# checking all the csv files in the 
# specified path

for filename in files:
    #la clave del diccionario es el nombre del archivo sin toda la ruta y el resultado es el propiodataset
    diccionario[filename[(len(path)+1):len(filename)-4]]=pd.read_csv(filename, index_col=None,on_bad_lines='skip',delimiter=";", decimal=",")
  
import matplotlib.pyplot as plt

# def imprimir_todo(lista):

#     for key in lista:
#         dataset=diccionario.get(key)
#         rms_gemelo_interno_derecho=dataset.iloc[:,7].values
#         eje_x=dataset.iloc[:,6].values
        
#         print(np.mean(rms_gemelo_interno_derecho))
#         plt.plot(eje_x,rms_gemelo_interno_derecho, label="rms_gemelo_interno_derecho")
#         # plt.plot(eje_x,np.mean(rms_gemelo_interno_derecho), label="media_gem_interno_der")
        
#         rms_tibial_anterior_derecho=dataset.iloc[:,8].values
#         plt.plot(eje_x,rms_tibial_anterior_derecho, label="rms_tibial_anterior_derecho")
    
#         rms_gemelo_interno_izquierdo=dataset.iloc[:,9].values
#         plt.plot(eje_x,rms_gemelo_interno_izquierdo, label="rms_gemelo_interno_izquierdo")
    
        
#         rms_tibial_anterior_izquierdo=dataset.iloc[:,10].values
#         plt.plot(eje_x,rms_tibial_anterior_izquierdo, label="rms_tibial_anterior_izquierdo")
        
#         plt.title(key)
#         plt.legend()
        
#         plt.figure()


#imprimir_todo(diccionario.keys())
    

#como en el otro excel Elvira recibe una puntuación de 1 y maria dolores puig de 11, voy a comparar sus gráficas sin la ayuda del andador.

#imprimir_todo(["maria_dolores_puig_baja0","petra_sanz_baja0","maria_dolores_puig_media0","petra_sanz_media0","maria_dolores_puig_alta0","petra_sanz_alta0"])  
    



# #simular el txt de bitalino
# for key in diccionario.keys():
#    dataset=diccionario.get(key)
#    rms_gemelo_interno_derecho=dataset.iloc[:,7].values
#    f = open (str(key)+"rms_gemelo_interno_derecho"+".txt",'w')
#    f.write("#"+key+"\n")
#    f.write("#rms_gemelo_interno_derecho\n")
#    stringtotal=""
#    for i in np.arange(len(dataset.iloc[:,6].values)):

#         string=str((i % 16))+"\t 0"+"\t 0"+"\t 0"+"\t 0"+"\t"+str(rms_gemelo_interno_derecho[i])+"\n"
#         stringtotal+=string
    

#    f.write(stringtotal)
#    f.close()
       
       
#simular el txt de bitalino
for key in diccionario.keys():
   dataset=diccionario.get(key)
   rms_tibial_anterior_derecho=dataset.iloc[:,8].values
   f = open (str(key)+"rms_tibial_anterior_derecho"+".txt",'w')
   f.write("#"+key+"\n")
   f.write("#rms_tibial_anterior_derecho\n")
   stringtotal=""
   for i in np.arange(len(dataset.iloc[:,6].values)):

        string=str((i % 16))+"\t 0"+"\t 0"+"\t 0"+"\t 0"+"\t"+str(rms_tibial_anterior_derecho[i])+"\n"
        stringtotal+=string
    

   f.write(stringtotal)
   f.close()
       
   
#simular el txt de bitalino
for key in diccionario.keys():
   dataset=diccionario.get(key)
   rms_gemelo_interno_izquierdo=dataset.iloc[:,9].values
   f = open (str(key)+"rms_gemelo_interno_izquierdo"+".txt",'w')
   f.write("#"+key+"\n")
   f.write("#rms_gemelo_interno_izquierdo\n")
   stringtotal=""
   for i in np.arange(len(dataset.iloc[:,9].values)):

        string=str((i % 16))+"\t 0"+"\t 0"+"\t 0"+"\t 0"+"\t"+str(rms_gemelo_interno_izquierdo[i])+"\n"
        stringtotal+=string
    

   f.write(stringtotal)
   f.close()
   
   
   
   #simular el txt de bitalino
   for key in diccionario.keys():
      dataset=diccionario.get(key)
      rms_tibial_anterior_izquierdo=dataset.iloc[:,10].values
      f = open (str(key)+"rms_tibial_anterior_izquierdo"+".txt",'w')
      f.write("#"+key+"\n")
      f.write("#rms_tibial_anterior_izquierdo\n")
      stringtotal=""
      for i in np.arange(len(dataset.iloc[:,10].values)):

           string=str((i % 16))+"\t 0"+"\t 0"+"\t 0"+"\t 0"+"\t"+str(rms_tibial_anterior_izquierdo[i])+"\n"
           stringtotal+=string
       

      f.write(stringtotal)
      f.close()
          
       
       
   # plt.plot(eje_x,rms_gemelo_interno_derecho, label="rms_gemelo_interno_derecho")
   
   # rms_tibial_anterior_derecho=dataset.iloc[:,8].values
   # plt.plot(eje_x,rms_tibial_anterior_derecho, label="rms_tibial_anterior_derecho")
     
   # rms_gemelo_interno_izquierdo=dataset.iloc[:,9].values
   # plt.plot(eje_x,rms_gemelo_interno_izquierdo, label="rms_gemelo_interno_izquierdo")
   
       
   # rms_tibial_anterior_izquierdo=dataset.iloc[:,10].values
   # plt.plot(eje_x,rms_tibial_anterior_izquierdo, label="rms_tibial_anterior_izquierdo")
   
   # plt.title(key)
   # plt.legend()          
   # plt.figure()
    
    
    
    
    
    
    
    
    
    
    

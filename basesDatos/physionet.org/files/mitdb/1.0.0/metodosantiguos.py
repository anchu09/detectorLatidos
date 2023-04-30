#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:22:53 2023

@author: dani
"""

#window para un unico archivo junto

def window(bandwidth):
    #me dan la bandwith en seegundos pero la pasamos a muestras
    muestras=bandwidth*360
    muestras=round(muestras)
    lista=diccionarioAnotacionesCeros.keys()
    with open("./con_window/archivoDatos.txt", "a") as archivoDatos:

        
        archivoDatos.write("'"+str(list(lista)[0])+"', ")
    
    archivoDatos.close()
    
    
    #juntar ecg
    # for k in lista:
        # with open("./con_window/todojuntoECG.csv", "a") as archivoCardio:
        #     # archivoCardio.write("\n")

        #     # archivoCardio.write("la key es " + str(key))
        #     # archivoCardio.write("\n")
            
        #     diccionarioDatos[key].to_csv(archivoCardio, header=False, index=False)
        #     archivoCardio.flush()

        #     archivoCardio.close()
    
    #juntar anotaciones
    with open("./con_window/todojuntoAnotaciones.csv", "a") as archivo:

        for key in diccionarioAnotacionesCeros.keys():
            #sacamos la columna 1 de toda la matriz de anotaciones originales en una lista
            print(key)
            # archivo.write("\nla key"+str(key)+"\n")
            listaMuestras=[]        
            for fila in diccionarioAnotaciones[key]:
                listaMuestras.append(fila[1])
            #ya tenemos el vector de las filas que hay que copiar
           
            for valor in listaMuestras:
                if valor==0:
                    valor=1
                fila_a_copiar = diccionarioAnotacionesCeros[key].loc[valor-1,diccionarioAnotacionesCeros[key].columns[2:]]
                
                if valor-muestras<0:
                    valor_menos_muestras=0
                else:
                    valor_menos_muestras=valor-muestras
                    
                if valor+muestras>650000:
                    valor_mas_muestras=650000
                
                else:
                    valor_mas_muestras=valor+muestras
                
                for muestraVentana in np.arange(valor_menos_muestras,valor_mas_muestras):
                    
    
                    diccionarioAnotacionesCeros[key].loc[muestraVentana-1,diccionarioAnotacionesCeros[key].columns[2:]]=fila_a_copiar
            
            diccionarioAnotacionesCeros[key].to_csv(archivo, index=False, header=False)
    #borrar lineas del tipo ,,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 a mano
    #ojo quitar la linea 21449968
    #comprobar que tiene el mismo numero de lineas que el otro archivo
    

    archivo.close()


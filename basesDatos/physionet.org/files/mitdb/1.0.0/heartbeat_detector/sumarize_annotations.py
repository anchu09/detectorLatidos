path = 'anotacionesCompletasCeros'
import os
import numpy as np
import pandas as pd
import glob
os.chdir(os.getcwd()[:-len("scientificProject")])
diccionarioAnotacionesCeros={}
path= 'leiblesDAT'

files = glob.glob(path + "/*.csv")


diccionarioDatos={}

for filename in files:
    diccionarioDatos[filename[(len(path)+1):len(filename)-4]]=pd.read_csv(filename,index_col=None,on_bad_lines='skip', delimiter="\t", header=None)






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


ruta_principal = './con_window_separado_resumido'
def window_separado(bandwidth):
    muestras=bandwidth*360
    muestras=round(muestras)
    for key in diccionarioAnotacionesCeros.keys():


        #ya tenemos el vector de las filas que hay que copiar
        for fila in np.arange(len(diccionarioAnotacionesCeros[key])):
            if (diccionarioAnotacionesCeros[key].iloc[fila, -4]=='N'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='L'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='R'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='B'):
                diccionarioAnotacionesCeros[key].iloc[fila, -4]='N'

            elif (diccionarioAnotacionesCeros[key].iloc[fila, -4]=='a'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='J'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='A'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='S'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='j'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='e'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='n'):
                diccionarioAnotacionesCeros[key].iloc[fila, -4]='S'

            elif (diccionarioAnotacionesCeros[key].iloc[fila, -4]=='V'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='E'):
                diccionarioAnotacionesCeros[key].iloc[fila, -4]='V'

            elif (diccionarioAnotacionesCeros[key].iloc[fila, -4]=='/'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='f'or diccionarioAnotacionesCeros[key].iloc[fila, -4]=='Q'):
                diccionarioAnotacionesCeros[key].iloc[fila, -4]='Q'
            elif (diccionarioAnotacionesCeros[key].iloc[fila, -4]=='F'):
                diccionarioAnotacionesCeros[key].iloc[fila, -4]='F'
            else:
                diccionarioAnotacionesCeros[key].iloc[fila, -4] = 'Z'


        diccionarioAnotacionesCeros[key].iloc[:,:].to_csv("./con_window_separado_resumido/"+key+".csv", index=False, header=False,sep=' ')


def window():

    path= './con_window_separado_resumido'

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

window()

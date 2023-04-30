#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 00:34:34 2023

@author: dani
"""

import os
import shutil

ruta = "/home/dani/tfg/basesDatos/physionet.org/files/mitdb/1.0.0/senales_troceadas/anotaciones"

# Obtener una lista de las carpetas en la ruta
carpetas = [nombre for nombre in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, nombre))]

# Iterar sobre cada carpeta y eliminar su contenido
for carpeta in carpetas:
    ruta_carpeta = os.path.join(ruta, carpeta)
    shutil.rmtree(ruta_carpeta)
    # Volver a crear la carpeta vacía
    os.mkdir(ruta_carpeta)

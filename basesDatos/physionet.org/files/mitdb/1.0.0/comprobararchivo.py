#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:00:38 2023

@author: dani
"""

import os
import numpy as np

directory = "./con_window_separado/"

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        data = np.loadtxt(filepath, delimiter=",")
        print(f"Archivo leído: {filename}")
        print(f"Datos: {data}")
    else:
        continue
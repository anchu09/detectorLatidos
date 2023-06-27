#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:01:42 2023

@author: dani
"""

import numpy as np
def segundos_a_segundos_minutos_y_horas(segundos):
    horas = int(segundos / 60 / 60)
    segundos -= horas * 60 * 60
    minutos = int(segundos / 60)
    segundos -= minutos * 60

    if (segundos >= 10):

        secs = str(round(segundos, 3)).ljust(6, '0')
    else:
        secs = str(round(segundos, 3)).ljust(5, '0')

    return str(minutos) + ":" + str(secs).zfill(6)


def cadenaEspacios(tamtiempo):
    cantidadespacios = 12 - tamtiempo
    cadena = ""
    for i in range(cantidadespacios):
        cadena += " "
    return cadena


def to_seconds(time_str):
    minutes, seconds_and_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_and_milliseconds.split('.')
    return int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def define_ma(L):
    return np.repeat(1. / (2 * L + 1), 2 * L + 1)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:25:39 2022

@author: dani
"""

#gene expression can be leaky

import numpy as np
import bokeh.io
import bokeh.plotting

bokeh.io.output_notebook()
R=np.linspace(0,20,200)
b0=1
kd=1
a0=0.25

beta= a0+b0/(1+R/kd)

p=bokeh.plotting.figure(
height=275,
width=400,
x_axis_label="R",
y_axis_label="B(R)",
x_range=[R[0],R[-1]],
y_range=[0,beta.max()],
)

p.line(R,beta,line_width=2, color="tomato", legend_label="B(R)")
p.line(
[R[0],R[-1]],
    [a0,a0],
    line_width=2,
    color="orange",
    legend_label="bassal expression",
)
p.title.text="kd=1, b0=1, a0=0.25"
bokeh.io.show(p)

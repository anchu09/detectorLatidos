#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:03:19 2023

@author: dani
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:27:37 2022

@author: dani
"""

import numpy as np
from keras.utils import Sequence
import randomTransformations


class CustomDataGenerator(Sequence):
    def __init__(self, x_filenames, y_filenames, batch_size, train):
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size
        self.train = train

    def __len__(self):
        return int(np.ceil(len(self.x_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_filenames = self.x_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_filenames = self.y_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = np.asarray([np.loadtxt(filename) for filename in batch_x_filenames]).astype(np.float32)

        batch_y = np.asarray([np.loadtxt(filename, delimiter=',') for filename in batch_y_filenames]).astype(np.int32)
        if self.train == True:
            random_number = np.random.randint(0, 7)  # devuelve hasta el 5

            batch_x, batch_y = randomTransformations.randomTransformation(random_number, batch_x, batch_y, titulo=True)

        return batch_x, batch_y





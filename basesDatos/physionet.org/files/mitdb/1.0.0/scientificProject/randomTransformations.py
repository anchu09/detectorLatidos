import numpy as np
import scipy.signal as sig
fs=360

def randomTransformation(random_number,batch_x,batch_y):
    if random_number == 0:
        print("no hago transformacion")

    elif random_number == 1:
        # print("pasobajo")
        batch_x = low_pass(batch_x)
    elif random_number == 2:
        print("pasoalto")
        batch_x = high_pass(batch_x)
    elif random_number == 3:
        print("bandpass")
        batch_x = band_pass(batch_x)
    elif random_number == 4:
        print("baseline wander")
    elif random_number == 5:
        print("ruido 50 hz")

    elif random_number == 6:
        print("añadir ruido con un srn aleatorio")

    elif random_number == 7:
        print("desplazamiento aleatorio")

    return batch_x,batch_y



def low_pass(batch_x):

    fcs = np.array([11]) / (fs / 2.)
    # b = sig.firwin(61, fcs, window='hamming', pass_zero=True)
    iir_b, iir_a = sig.butter(6, fcs, 'lowpass')

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x


def high_pass(batch_x):

    fcs = np.array([5]) / (fs / 2.)

    iir_b, iir_a = sig.butter(6, fcs, 'highpass')

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x

def band_pass(batch_x):

    fcs = np.array([5,11]) / (fs / 2.)

    iir_b, iir_a = sig.butter(6, fcs, 'bandpass')

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import random
fs=360

def randomTransformation(random_number,batch_x,batch_y, titulo):
    if random_number == 0:
        plt.title("no hago transformacion")

    elif random_number == 1:
        batch_x = low_pass(batch_x, titulo)

    elif random_number == 2:
        batch_x = high_pass(batch_x, titulo)
    elif random_number == 3:
        batch_x = band_pass(batch_x, titulo)
    elif random_number == 4:
        batch_x=add_baseline_wander(batch_x, titulo)


    elif random_number == 5:
        batch_x=add_50hz(batch_x, titulo)

    elif random_number == 6:
        batch_x = add_noise(batch_x, titulo)


    elif random_number == 7:
        print("desplazamiento aleatorio")

    return batch_x,batch_y



def low_pass(batch_x, titulo):
    fc=random.uniform(30,100)

    fcs = np.array([fc]) / (fs / 2.)
    # b = sig.firwin(61, fcs, window='hamming', pass_zero=True)
    iir_b, iir_a = sig.butter(6, fcs, 'lowpass')
    if titulo==True:
        plt.title("pasobajo" + str(fc))

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x


def high_pass(batch_x, titulo):
    fc=random.uniform(0.5,1)
    if titulo==True:

        plt.title("pasoalto"+str(fc))

    fcs = np.array([fc]) / (fs / 2.)

    iir_b, iir_a = sig.butter(6, fcs, 'highpass')

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x

def band_pass(batch_x, titulo):
    fc1=random.uniform(0.5,1)
    fc2=random.uniform(30,100)
    if titulo==True:

        plt.title("bandpass"+str(fc1)+" - "+ str(fc2))

    fcs = np.array([fc1,fc2]) / (fs / 2.)

    iir_b, iir_a = sig.butter(3, fcs, 'bandpass')

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x

def add_noise(batch_x, titulo, snr_min=100,snr_max=140):
    snr = np.random.uniform(snr_min, snr_max)
    # snr=110
    signal_power1 = np.mean(batch_x[0,:,0] ** 2)
    signal_power2 = np.mean(batch_x[0,:,1] ** 2)
    if titulo==True:

        plt.title("añadir ruido con un srn aleatorio"+str(snr))

    noise_power1 = signal_power1 / snr
    noise_power2 = signal_power2 / snr

    noise1 = np.random.normal(0, np.sqrt(noise_power1), len(batch_x[0,:,0]))
    noise2=np.random.normal(0, np.sqrt(noise_power2), len(batch_x[0,:,1]))
    batch_x[0,:,0] = batch_x[0,:,0] + noise1
    batch_x[0,:,1] = batch_x[0,:,1] + noise2

    return batch_x

def add_baseline_wander(batch_x, titulo):
    max_amplitude0=np.mean(abs(batch_x[0, :, 0]))
    max_amplitude1=np.mean(abs(batch_x[0, :, 1]))


    amplitude0 = np.random.uniform(0, max_amplitude0)
    amplitude1 = np.random.uniform(0, max_amplitude1)

    fbaseline=np.random.uniform(0.05, 0.5)
    # print(fbaseline)
    time = np.arange(len(batch_x[0, :, 1])) / 360
    if titulo==True:

        plt.title("baseline wander"+str(max_amplitude0)+" fbas "+str(fbaseline))
    baseline_wander0 = amplitude0*np.sin(2 * np.pi * fbaseline * time)
    baseline_wander1 = amplitude1*np.sin(2 * np.pi * fbaseline * time)


    batch_x[0, :, 0]=batch_x[0,:,0]+baseline_wander0
    batch_x[0, :, 1]=batch_x[0,:,1]+baseline_wander1

    return batch_x

def add_50hz(batch_x, titulo, snr_min=100,snr_max=140):
    # Genera un valor aleatorio de SNR dentro del rango especificado
    snr = np.random.uniform(snr_min, snr_max)
    # snr=100
    signal_power0 = np.mean(batch_x[0,:,0] ** 2)
    signal_power1 = np.mean(batch_x[0,:,1] ** 2)

    noise_power0 = signal_power0 / snr
    noise_power1 = signal_power1 / snr

    t = np.arange(len(batch_x[0, :, 0]))/360
    noise = np.sin(2 * np.pi * 50 * t)



    harmonico_max = np.random.randint(1,5)
    harmonicos = np.random.choice(range(2, 6), harmonico_max, replace=False)
    harmonicos=np.sort(harmonicos)

    # print(harmonico_max)
    # print(harmonicos)
    if titulo==True:

        plt.title("harmonicos" +str(harmonicos))
    for h in harmonicos:
        noise += np.sin(2 * np.pi * 50 * h * t)

    noise0 = np.sqrt(noise_power0) * noise
    noise1 = np.sqrt(noise_power1) * noise

    batch_x[0, :, 0] = batch_x[0, :, 0] + noise0
    batch_x[0, :, 1] = batch_x[0, :, 1] + noise1

    return batch_x
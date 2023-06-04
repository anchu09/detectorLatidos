import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import random
fs=360

def randomTransformation(random_number,batch_x,batch_y):
    if random_number == 0:
        print("no hago transformacion")

    elif random_number == 1:
        # print("pasobajo")
        batch_x = low_pass(batch_x)
    elif random_number == 2:
        # print("pasoalto")
        batch_x = high_pass(batch_x)
    elif random_number == 3:
        # print("bandpass")
        batch_x = band_pass(batch_x)
    elif random_number == 4:
        # print("baseline wander")
        batch_x=add_baseline_wander(batch_x)


    elif random_number == 5:
        # print("ruido 50 hz")
        batch_x=add_50hz(batch_x)

    elif random_number == 6:
        # print("añadir ruido con un srn aleatorio")
        batch_x = add_noise(batch_x)


    elif random_number == 7:
        print("desplazamiento aleatorio")

    return batch_x,batch_y



def low_pass(batch_x):
    fc=random.uniform(30,100)

    fcs = np.array([fc]) / (fs / 2.)
    # b = sig.firwin(61, fcs, window='hamming', pass_zero=True)
    iir_b, iir_a = sig.butter(6, fcs, 'lowpass')

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x


def high_pass(batch_x):
    fc=random.uniform(0.5,1)

    fcs = np.array([fc]) / (fs / 2.)

    iir_b, iir_a = sig.butter(6, fcs, 'highpass')

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x

def band_pass(batch_x):
    fc1=random.uniform(0.5,1)
    fc2=random.uniform(30,100)
    fc1=1
    fc2=30
    fcs = np.array([fc1,fc2]) / (fs / 2.)

    iir_b, iir_a = sig.butter(3, fcs, 'bandpass')

    for i in np.arange(batch_x.shape[2]):

        batch_x[0,:,i]=sig.filtfilt(iir_b, iir_a, batch_x[0,:,i])

    return batch_x

def add_noise(batch_x, snr_min=100,snr_max=120):
    snr = np.random.uniform(snr_min, snr_max)
    # snr=110
    signal_power1 = np.mean(batch_x[0,:,0] ** 2)
    signal_power2 = np.mean(batch_x[0,:,1] ** 2)

    noise_power1 = signal_power1 / snr
    noise_power2 = signal_power2 / snr

    noise1 = np.random.normal(0, np.sqrt(noise_power1), len(batch_x[0,:,0]))
    noise2=np.random.normal(0, np.sqrt(noise_power2), len(batch_x[0,:,1]))
    batch_x[0,:,0] = batch_x[0,:,0] + noise1
    batch_x[0,:,1] = batch_x[0,:,1] + noise2

    return batch_x

def add_baseline_wander(batch_x):
    max_amplitude0=max(batch_x[0, :, 0])
    max_amplitude1=max(batch_x[0, :, 1])


    amplitude0 = np.random.uniform(0, max_amplitude0)
    amplitude1 = np.random.uniform(0, max_amplitude1)

    fbaseline=np.random.uniform(0.05, 0.5)

    time = np.arange(len(batch_x[0, :, 1])) / 360

    baseline_wander0 = amplitude0*np.sin(2 * np.pi * fbaseline * time)
    baseline_wander1 = amplitude1*np.sin(2 * np.pi * fbaseline * time)


    batch_x[0, :, 0]=batch_x[0,:,0]+baseline_wander0
    batch_x[0, :, 1]=batch_x[0,:,1]+baseline_wander1

    return batch_x

def add_50hz(batch_x, snr_min=5,snr_max=20):
    # Genera un valor aleatorio de SNR dentro del rango especificado
    snr = np.random.uniform(snr_min, snr_max)
    snr=100
    signal_power0 = np.mean(batch_x[0,:,0] ** 2)
    signal_power1 = np.mean(batch_x[0,:,1] ** 2)

    noise_power0 = signal_power0 / snr
    noise_power1 = signal_power1 / snr

    t = np.arange(len(batch_x[0, :, 0]))
    noise = np.sin(2 * np.pi * 50 * t / len(batch_x[0, :, 0]))



    harmonico_max = np.random.uniform(2, 6)
    # harmonicos = [2, 3, 4, 5]
    harmonicos=np.arange(2,int(harmonico_max)+1)
    # print(harmonico_max)
    # print(harmonicos)
    plt.title("harmonicos" +str(harmonicos))
    for h in harmonicos:
        noise += np.sin(2 * np.pi * 50 * h * t / len(batch_x[0, :, 0]))

    noise0 = np.sqrt(noise_power0) * noise
    noise1 = np.sqrt(noise_power1) * noise

    batch_x[0, :, 0] = batch_x[0, :, 0] + noise0
    batch_x[0, :, 1] = batch_x[0, :, 1] + noise1

    return batch_x
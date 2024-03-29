""" 3 functions for the LISA noise:
    1: LISA Noise only
    2: LISA Noise and DWD Noise Boileau et al.
    3: LISA Noise and DWD Noise Robson & Cornish
     
    The noise parameters are given the values estimated in the literature """

import numpy as np
import matplotlib.pyplot as plt  
from giese_lisa_sens import Omega_N
from ps_rewritten import PowerSpectrum
from noise import lisa_noise_1, lisa_noise_2
import time

def make_data_no_DWD(frequencies, N_c, PS):
    def calculate_N():
        total_noise = np.zeros((len(frequencies), N_c))
                
        for i, f in enumerate(frequencies):
            for j in range(N_c):
                G_1 = np.random.normal(loc=0.0, scale=np.sqrt(Omega_N(f, 3, 15)))
                G_2 = np.random.normal(loc=0.0, scale=np.sqrt(Omega_N(f, 3, 15)))
                N = (G_1**2 + G_2**2) / 2
                total_noise[i, j] = N
        return total_noise

    def calculate_S():
        total_signal = np.zeros((len(frequencies), N_c))  # Initialize a 2D array
        
        for i, f in enumerate(frequencies):
            for j in range(N_c):
                G_1 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(f, PS.Amp, PS.fp_0())))
                G_2 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(f, PS.Amp, PS.fp_0())))
                S = (G_1**2 + G_2**2) / 2
                total_signal[i, j] = S
        return total_signal

    # Calculate noise and signal arrays and adding them together to form the data
    Total_Noise = calculate_N()
    Total_Signal = calculate_S()
    Total_Data = (Total_Noise + Total_Signal)
    return Total_Data

def make_data_DWD_1(frequencies, N_c, PS):
    def calculate_N():
        total_noise = np.zeros((len(frequencies), N_c))
                
        for i, f in enumerate(frequencies):
            for j in range(N_c):
                G_1 = np.random.normal(loc=0.0, scale=np.sqrt(lisa_noise_1(f, 7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15)))
                G_2 = np.random.normal(loc=0.0, scale=np.sqrt(lisa_noise_1(f, 7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15)))
                N = (G_1**2 + G_2**2) / 2
                total_noise[i, j] = N
        return total_noise

    def calculate_S():
        total_signal = np.zeros((len(frequencies), N_c))  # Initialize a 2D array
        
        for i, f in enumerate(frequencies):
            for j in range(N_c):
                G_1 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(f, PS.Amp, PS.fp_0())))
                G_2 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(f, PS.Amp, PS.fp_0())))
                S = (G_1**2 + G_2**2) / 2
                total_signal[i, j] = S
        return total_signal

    # Calculate noise and signal arrays and adding them together to form the data
    Total_Noise = calculate_N()
    Total_Signal = calculate_S()
    Total_Data = (Total_Noise + Total_Signal)
    return Total_Data

def make_data_DWD_2(frequencies, N_c, PS):
    def calculate_N():
        total_noise = np.zeros((len(frequencies), N_c))
                
        for i, f in enumerate(frequencies):
            for j in range(N_c):
                G_1 = np.random.normal(loc=0.0, scale=np.sqrt(lisa_noise_2(f, 0.138, -221, 521, 1680, 3, 15)))
                G_2 = np.random.normal(loc=0.0, scale=np.sqrt(lisa_noise_2(f, 0.138, -221, 521, 1680, 3, 15)))
                N = (G_1**2 + G_2**2) / 2
                total_noise[i, j] = N
        return total_noise

    def calculate_S():
        total_signal = np.zeros((len(frequencies), N_c))  # Initialize a 2D array
        
        for i, f in enumerate(frequencies):
            for j in range(N_c):
                G_1 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(f, PS.Amp, PS.fp_0())))
                G_2 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(f, PS.Amp, PS.fp_0())))
                S = (G_1**2 + G_2**2) / 2
                total_signal[i, j] = S
        return total_signal

    # Calculate noise and signal arrays and adding them together to form the data
    Total_Noise = calculate_N()
    Total_Signal = calculate_S()
    Total_Data = (Total_Noise + Total_Signal)
    return Total_Data

if __name__ == "__main__":
    start_time = time.time()
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high))
    PS = PowerSpectrum(0.4, 100, 180, 0.8)
    GW_model = PS.Omega_GW(frequencies, PS.Amp, PS.fp_0())
    DATA1 = make_data_no_DWD(frequencies, 94, PS)
    DATA2 = make_data_DWD_1(frequencies, 94, PS)
    DATA3 = make_data_DWD_2(frequencies, 94, PS)
    Sensitivity = Omega_N(frequencies, 3, 15)

    # Timing code
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"RunTime: {elapsed_time} seconds")


    plt.loglog(frequencies, Sensitivity, label='LISA Sensitvity', color='b')
    plt.loglog(frequencies, GW_model, label='GW Signal', color='k' )
    plt.loglog(frequencies, np.mean(DATA1, axis=1), label='LISA Noise', color='y')
    plt.loglog(frequencies, np.mean(DATA2, axis=1), label='DWD Noise 1', color='c')
    plt.loglog(frequencies, np.mean(DATA3, axis=1), label='DWD Noise 2', color='m')
    plt.title(r'$LISA$' + " " + r'$\Omega$' + " " + '$Mock Data Generation$')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.savefig('DWD Mock Data + LISA Sensitivity + GW Model')
    plt.legend() 
    plt.show()




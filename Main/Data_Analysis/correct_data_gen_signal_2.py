# Generating both Noise and Signal Data
import numpy as np
import matplotlib.pyplot as plt  
from giese_lisa_sens import S_n, P_acc, P_oms, Omega_N
from ps_rewritten import PowerSpectrum
import time

# Calculating S_i


def calculate_S(frequencies, N_c, PS, Amp, f_peak):
    total_signal = np.zeros((len(frequencies), N_c))  # Initialize a 2D array
        
    for i, f in enumerate(frequencies):
        for j in range(N_c):
            G_1 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(f, Amp, f_peak)))
            G_2 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(f, Amp, f_peak)))
            S = (G_1**2 + G_2**2) / 2
            total_signal[i, j] = S

    return total_signal

if __name__ == "__main__":
    start_time = time.time()
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high))
    PS = PowerSpectrum(0.6, 50, 180, 0.8)
    GW = PS.Omega_GW(frequencies, PS.Amp, PS.fp_0())
    print(GW.shape)
    Amp = PS.Amp
    print("The amplitude is:", Amp)
    f_peak = PS.fp_0()
    print("The peak frequency is:", f_peak)
    N_c = 50 # Number of datapoints to be generated for each f
    NOISE = calculate_S(frequencies, N_c, PS, Amp, f_peak)


    # Want to obtain the mean and st. dev for each frequency over N_c points
    # Calculating mean for each f
    mean_values = np.mean(NOISE, axis=1)
    print("Shape of mean_values array:", mean_values.shape)
    print("Mean values array:", mean_values)

    # Calculating standard deviation for each f
    std_dev_values = np.std(NOISE, axis=1)
    print("Shape of st_dev_values array:", std_dev_values.shape)
    print("Standard Deviation Values array:", std_dev_values)
    end_time = time.time()
    total_t = end_time - start_time
    print("Run time:", total_t, "s")

    # Plotting sample and model data
    plt.loglog(frequencies, mean_values, color='blue', label="Sample Data") # Sample data
    plt.loglog(frequencies, Omega_N(frequencies, 3, 15), color='red', label="Model Data") # Model data
    plt.legend()
    plt.title(r'Sample vs Model Data Values for LISA Noise Signal')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.grid = 'True'
    plt.savefig("Sample_vs_Model_Noise")
    plt.show() 
import numpy as np
import matplotlib.pyplot as plt
from giese_lisa_sens import Omega_N
from ps import PowerSpectrum

def make_data(frequencies, N_c, A, P, PS):
    Total_Data = []

    for _ in range(N_c):
        Total_Noise = np.empty(0)
        Total_Signal = np.empty(0)

        for f in frequencies:
            # Calculate noise for each frequency
            GW_noise = Omega_N(f, A, P)
            st_dev_noise = np.sqrt(GW_noise)
            
            # Generate random numbers for noise
            G_noise_1 = np.random.normal(loc=0.0, scale=st_dev_noise)
            G_noise_2 = np.random.normal(loc=0.0, scale=st_dev_noise)
            N = (G_noise_1**2 + G_noise_2**2) / 2
            Total_Noise = np.append(Total_Noise, N)

            # Calculate signal for each frequency
            GW_signal = PS.Omega_GW(f)
            st_dev_signal = np.sqrt(GW_signal)

            # Generate random numbers for signal
            G_signal_1 = np.random.normal(loc=0.0, scale=st_dev_signal)
            G_signal_2 = np.random.normal(loc=0.0, scale=st_dev_signal)
            S = (G_signal_1**2 + G_signal_2**2) / 2
            Total_Signal = np.append(Total_Signal, S)

        # Sum noise and signal for each frequency and calculate the mean
        Total_Data.append(np.mean(Total_Noise + Total_Signal))

    # Convert the list of means to a numpy array
    Total_Data = np.array(Total_Data)

    return Total_Data


if __name__ == "__main__":
    #f_low = np.arange(0.00003, 0.001, 0.000001)
    #f_middle = np.arange(0.001, 0.01, 0.00001)
    #f_high = np.arange(0.01, 0.5, 0.0005)
    #frequencies = np.concatenate((f_low, f_middle, f_high))
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high))
    print(frequencies.shape)
    P1 = PowerSpectrum(0.6, 50, 180, 0.8)
    GW_model = P1.Omega_GW(frequencies)
    DATA = make_data(frequencies, 2, 3, 15, P1)
    print(DATA.shape)

    # Plotting sample and model data
    plt.loglog(frequencies, DATA, color='blue', label="Sample Data") # Sample data
    plt.loglog(frequencies, GW_model, color='red', label="Model Signal")
    plt.loglog(frequencies, Omega_N(frequencies, 3, 15), color='green', label="Model Noise") # Model Noise data
    plt.legend()
    plt.title(r'Sample vs Model Data Values for LISA Noise Signal')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.grid = 'True'
    plt.savefig("Sample_vs_Model_Noise")
    plt.show()

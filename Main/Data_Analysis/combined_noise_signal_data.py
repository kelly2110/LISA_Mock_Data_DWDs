import numpy as np
import matplotlib.pyplot as plt  
from giese_lisa_sens import S_n, P_acc, P_oms, Omega_N
from ps import PowerSpectrum
import time


def make_data(frequencies, N_c, A, P, PS):
    def calculate_N(frequencies, N_c, A, P):
        total_noise = []

        for _ in range(N_c):
                G_1 = np.random.normal(loc=0.0, scale=np.sqrt(Omega_N(frequencies, A, P)))
                G_2 = np.random.normal(loc=0.0, scale=np.sqrt(Omega_N(frequencies, A, P)))
                N = (G_1**2 + G_2**2) / 2
                total_noise.append(N)
                print("Total noise:", total_noise[0:2])

        return np.array(total_noise)

    def calculate_S(frequencies, N_c, PS):
        total_signal = []
    
        for _ in range(N_c):
                G_1 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(frequencies)))
                G_2 = np.random.normal(loc=0.0, scale=np.sqrt(PS.Omega_GW(frequencies)))
                S = (G_1**2 + G_2**2) / 2
                total_signal.append(S)
                print("Total Signal:", total_signal[0:2])

        return np.array(total_signal)

    # Calculate noise and signal arrays and adding them together to form the data
    Total_Noise = calculate_N(frequencies, N_c, A, P)
    Total_Signal = calculate_S(frequencies, N_c, PS)
    Total_Data = np.array(Total_Noise + Total_Signal)
    return Total_Data


if __name__ == "__main__":
    start_time = time.time()
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high))
    P1 = PowerSpectrum(0.6, 50, 180, 0.8)
    GW_model = P1.Omega_GW(frequencies)
    DATA = make_data(frequencies, 5, 3, 15, P1)
    print("Data:",DATA[0:2])
    mean = np.mean(DATA, axis=0)
    std_dev = np.std(DATA, axis=0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"RunTime: {elapsed_time} seconds")


    # Plotting sample and model data
    plt.loglog(frequencies, mean, color='blue', label="Sample Data") # Sample data
    plt.loglog(frequencies, GW_model, color='red', label="Model Signal")
    plt.loglog(frequencies, Omega_N(frequencies, 3, 15), color='green', label="Model Noise") # Model Noise data
    plt.legend()
    plt.title(r'Sample vs Model Data Values for LISA Noise Signal')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.grid = 'True'
    plt.savefig("Sample_vs_Model_Noise_1")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from giese_lisa_sens import Omega_N
from ps import PowerSpectrum

def make_data(frequencies, N_c, A, P, PS):
    for _ in range(N_c):
        for f in frequencies:
            Total_Noise = np.empty
            Total_Signal = np.empty
            Total_Data = np.empty

            Noise_f = []
            Signal_f = []

            GW = Omega_N(f, A, P)
            st_dev = np.sqrt(GW)
            G_1 = np.random.normal(loc=0.0, scale=st_dev)
            G_2 = np.random.normal(loc=0.0, scale=st_dev)
            N = (G_1**2 + G_2**2) / 2
            Noise_f.append(N)
            Total_Noise = np.array((Noise_f))

            GW_s = np.array(PS.Omega_GW(f))
            st_dev_all = np.sqrt(GW_s)
            G_1_all = np.random.normal(loc=0.0, scale=st_dev_all)
            G_2_all = np.random.normal(loc=0.0, scale=st_dev_all)
            S_all = (G_1_all**2 + G_2_all**2) / 2
            Signal_f.append(S_all)
            Total_Signal = np.array((Signal_f))

        
            Total_Data = np.array((Total_Noise + Total_Signal))
            Mean_Data = np.mean(Total_Data)
            print(Mean_Data.shape)
            return Mean_Data



if __name__ == "__main__":
    #f_low = np.arange(0.00003, 0.001, 0.000001)
    #f_middle = np.arange(0.001, 0.01, 0.00001)
    #f_high = np.arange(0.01, 0.5, 0.0005)
    #frequencies = np.concatenate((f_low, f_middle, f_high))
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high))
    P1 = PowerSpectrum(0.6, 50, 180, 0.8)
    GW_model = P1.Omega_GW(frequencies)
    DATA = make_data(frequencies, 2, 3, 15, P1)


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

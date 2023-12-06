# Generating Noise Data Only

import numpy as np
import matplotlib.pyplot as plt  
from giese_lisa_sens import S_n, P_acc, P_oms, Omega_N
import time


# Calculating N_i
""" def calculate_N(frequencies, N_c, A, P):
    Total_Noise = []

    for _ in range(N_c):
        Noise_f = []
        for f in frequencies:
            GW = Omega_N(f, A, P)
            st_dev = np.sqrt(GW)
            # Generate random numbers from a Gaussian distribution with zero mean and the specified standard deviation
            G_1 = np.random.normal(loc=0.0, scale=st_dev)
            G_2 = np.random.normal(loc=0.0, scale=st_dev)
            N = (G_1**2 + G_2**2) / 2
            Noise_f.append(N)

        Total_Noise.append(np.array(Noise_f))

    return np.array(Total_Noise) """ 

""" def calculate_N(frequencies, N_c, A, P):
    total_noise = []

    for _ in range(N_c):
            G_1 = np.random.normal(loc=0.0, scale=np.sqrt(Omega_N(frequencies, A, P)))
            G_2 = np.random.normal(loc=0.0, scale=np.sqrt(Omega_N(frequencies, A, P)))
            N = (G_1**2 + G_2**2) / 2
            total_noise.append(N)

    return np.array(total_noise) """
def calculate_N(frequencies, N_c, A, P):
    total_noise = np.zeros((len(frequencies), N_c))
                
    for i, f in enumerate(frequencies):
        for j in range(N_c):
            G_1 = np.random.normal(loc=0.0, scale=np.sqrt(Omega_N(f, A, P)))
            G_2 = np.random.normal(loc=0.0, scale=np.sqrt(Omega_N(f, A, P)))
            N = (G_1**2 + G_2**2) / 2
            total_noise[i, j] = N
                    

    print(total_noise)
    return total_noise

if __name__ == "__main__":
    begin_t = time.time()
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high))
    N_c = 50 # Number of datapoints to be generated for each f
    NOISE = calculate_N(frequencies, N_c, 3, 15)

    print(np.size(NOISE))  # Array total size, dimension1xdimension2
    print(NOISE.shape)     # Actual array dimensions
    print("Number of datapoints for each frequency:", N_c )
    print("Number of frequencies:", frequencies.shape)
    
    # Want to obtain the mean and st. dev for each frequency over N_c points
    # Calculating mean for each f
    mean_values = np.mean(NOISE, axis=1)
    print("Mean values array:", mean_values)

    # Calculating standard deviation for each f
    std_dev_values = np.std(NOISE, axis=1)
    print("Standard Deviation Values array:", std_dev_values)
    end_time = time.time()
    total_t = end_time - begin_t
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




# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps_rewritten import PowerSpectrum
from combined_data_gen import make_data_DWD_1, make_data_DWD_2, make_data_no_DWD
from Main.Data_Analysis.SNR_calculation import calculate_snr_
import time

# Defining the chi-squared function
start_time = time.time()
def chi_squared(params, frequencies, powerspectrum, mean_sample_data, standard_deviation):
    chi_2 = []
    A, P, Amp, f_peak = params
    noise_model = Omega_N(frequencies, A, P)
    GW_model = powerspectrum.Omega_GW(frequencies, Amp, f_peak)

    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - GW_model - noise_model) / standard_deviation)**2)
    print(chi_2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    print(" The chi squared value is", chi_2_value)
    return chi_2_value

# Specifying frequency range and number of datapoints:
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
N_c = 50

# Generating the mock data, taking the mean and the standard deviation
powerspectrum = PowerSpectrum(0.1, 50, 180, 0.6)

DATA = make_data_no_DWD(frequencies, N_c, powerspectrum)
mean_sample_data = np.mean(DATA, axis=1) # Should probably include this in the function/class?
standard_deviation = np.std(DATA, axis=1)

# Initial parameter guess
initial_params = [3, 15, 1e-9, 1e-3] 

# Minimize the chi-squared function
result = minimize(chi_squared, initial_params, args=(frequencies, powerspectrum, mean_sample_data, standard_deviation), method='Powell', tol=1e-11)
Chi_Squared_bf = result.fun
print(f"Best fit Chi Squared value:", Chi_Squared_bf)

# Extracting the optimized parameters
optimized_A, optimized_P, optimized_Amp, optimized_f_p0 = result.x
print(f"Optimized Am: {optimized_A}")
print(f"Optimized P: {optimized_P}")
print(f"Optimized Amplitude: {optimized_Amp}")
print(f"Optimized Peak Frequency: {optimized_f_p0}")

# Convergence check
if result.success:   
    print("Optimization converged successfully.")
# Checking for convergence
else:
    print("Optimization did not converge.")


# Reconstructing the signal with optimized parameters
optimized_signal = powerspectrum.Omega_GW(frequencies, optimized_Amp, optimized_f_p0)

# Original signal
original_signal = powerspectrum.Omega_GW(frequencies, initial_params[2], initial_params[3])

# SNR Calculations 

# Original signal
SNR = calculate_snr_(original_signal, Omega_N(frequencies, 3, 15), frequencies)

# Reconstructed signal
SNR = calculate_snr_(optimized_signal, Omega_N(frequencies, 3, 15), frequencies)

# Plotting
plt.loglog(frequencies, mean_sample_data, label = 'Generated Mock Data', color='b')
plt.loglog(frequencies, original_signal, label='Original Signal', color='r')
plt.loglog(frequencies, optimized_signal, label='Reconstructed Signal', color='g')
plt.loglog(frequencies, Omega_N(frequencies, 3, 15), label='LISA Sensitivity', color='k')
plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
plt.ylabel(r'$h^{2}\Omega(f)$')
plt.title(r'Original and reconstructed gravitational wave signal')
plt.savefig('Original and reconstructed gravitational wave signal') 
plt.legend()
plt.show()

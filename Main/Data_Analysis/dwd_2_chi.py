# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps_rewritten import PowerSpectrum
from combined_data_gen import make_data_DWD_1, make_data_DWD_2, make_data_no_DWD
from noise import lisa_noise_1, lisa_noise_2
from SNR_calculation import calculate_snr_
import time

# Defining the chi-squared function
start_time = time.time()
def chi_squared(params, frequencies, powerspectrum, mean_sample_data, standard_deviation):
    chi_2 = []
    alpha, beta, gamma, k, A, P , Amp, f_peak = params
    noise_model = lisa_noise_2(frequencies, alpha, beta, gamma, k, A, P)
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
N_c = 25

# Generating the mock data, taking the mean and the standard deviation
powerspectrum = PowerSpectrum(0.45, 100, 180, 0.76)

DATA = make_data_DWD_2(frequencies, N_c, powerspectrum)
mean_sample_data = np.mean(DATA, axis=1) # Should probably include this in the function/class?
standard_deviation = np.std(DATA, axis=1)

# Initial parameter guess
initial_params = [0.138, -221, 1680, 521, 3, 15, 1e-9, 1e-3] 

# Minimize the chi-squared function
result = minimize(chi_squared, initial_params, args=(frequencies, powerspectrum, mean_sample_data, standard_deviation), method='Powell', tol=1e-20)
Chi_Squared_bf = result.fun
print(f"Best fit Chi Squared value:", Chi_Squared_bf)

# Extracting the optimized parameters
optimized_alpha, optimized_beta, optimized_gamma, optimized_k, optimized_A, optimized_P, optimized_Amp, optimized_f_p0 = result.x
print(f"Optimized alpha: {optimized_alpha}")
print(f"Optimized beta: {optimized_beta}")
print(f"Optimized gamma: {optimized_gamma}")
print(f"Optimized k: {optimized_k}")
print(f"Optimized A: {optimized_A}")
print(f"Optimized P: {optimized_P}")
print(f"Optimized Amplitude: {optimized_Amp}")
print(f"Optimized Peak Frequency: {optimized_f_p0}")

# Convergence check
if result.success:   
    print("Optimization converged successfully.")
# Checking for convergence
else:
    print("Optimization did not converge.")

# AIC calculation, adjust k as necessary
def calculate_aic(chi):
    k = 8
    AIC = chi + 2*k
    print("The AIC value is:", AIC)
    return AIC

calculate_aic(Chi_Squared_bf)

# Timing code snippet
end_time = time.time()
run_time = end_time - start_time
print(f"The run time is {run_time} seconds")

# Reconstructing the signal with optimized parameters
optimized_signal = powerspectrum.Omega_GW(frequencies, optimized_Amp, optimized_f_p0)

# Original signal
original_signal = powerspectrum.Omega_GW(frequencies, powerspectrum.Amp, powerspectrum.fp_0())

# SNR Calculations 

# Original signal
SNR = calculate_snr_(original_signal, Omega_N(frequencies, 3, 15), frequencies)

# Reconstructed signal
SNR = calculate_snr_(optimized_signal, Omega_N(frequencies, 3, 15), frequencies)

# Plotting
plt.loglog(frequencies, mean_sample_data, label = 'Generated Mock Data', color='b')
plt.loglog(frequencies, original_signal, label='Original Signal', color='c')
plt.loglog(frequencies, optimized_signal, label='Reconstructed Signal', color='g')
plt.loglog(frequencies, Omega_N(frequencies, 3, 15), label='LISA Sensitivity', color='k')
plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
plt.ylabel(r'$h^{2}\Omega(f)$')
plt.title(r'Original and reconstructed gravitational wave signal')
plt.savefig('Original and reconstructed gravitational wave signal') 
plt.legend()
plt.show()

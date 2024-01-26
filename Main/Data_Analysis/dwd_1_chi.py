# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps_rewritten import PowerSpectrum
from combined_data_gen import make_data_DWD_1, make_data_DWD_2, make_data_no_DWD
from noise import lisa_noise_1, lisa_noise_2
from SNR_calculation import calculate_snr_
from AIC import calculate_aic
import time

# Making and defining Data, this is all going to be automatized
# Specifying frequency range and number of datapoints:
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
N_c = 95

 # Generating the mock data, taking the mean and the standard deviation
powerspectrum = PowerSpectrum(0.45, 100, 180, 0.76)
DATA = make_data_DWD_1(frequencies, N_c, powerspectrum)
mean_sample_data = np.mean(DATA, axis=1)
standard_deviation = np.std(DATA, axis=1)

# Defining the chi-squared function
def chi_squared1(params, frequencies, powerspectrum, mean_sample_data, standard_deviation):
    chi_2 = []
    A1, A2, alpha1, alpha2, A, P , Amp, f_peak = params
    noise_model = lisa_noise_1(frequencies, A1, A2, alpha1, alpha2, A, P)
    GW_model = powerspectrum.Omega_GW(frequencies, Amp, f_peak)

    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - GW_model - noise_model) / standard_deviation)**2)
    print(chi_2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    print(" The chi squared value is", chi_2_value)
    return chi_2_value

# Defining the chi-squared function
def chi_squared(params, frequencies, powerspectrum, mean_sample_data, standard_deviation):
    chi_2 = []
    A1, A2, alpha1, alpha2, A, P = params
    noise_model = lisa_noise_1(frequencies, A1, A2, alpha1, alpha2, A, P)
    GW_model = powerspectrum.Omega_GW(frequencies, powerspectrum.Amp, powerspectrum.fp_0())

    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - GW_model - noise_model) / standard_deviation)**2)
    print(chi_2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    print(" The chi squared value is", chi_2_value)
    return chi_2_value

# Minimize the chi-squared function
initial_params1 = [7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15, powerspectrum.Amp, powerspectrum.fp_0()] 
initial_params= [7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15] 
result1 = minimize(chi_squared1, initial_params1, args=(frequencies, powerspectrum, mean_sample_data, standard_deviation), method='Powell', tol=1e-20)
result = minimize(chi_squared, initial_params, args=(frequencies, powerspectrum, mean_sample_data, standard_deviation), method='Powell', tol=1e-20)
Chi_Squared_bf = result1.fun
print(f"Best fit Chi Squared value:", Chi_Squared_bf)

# Extracting the optimized parameters
optimized_A1, optimized_A2, optimized_alpha1, optimized_alpha2, optimized_A, optimized_P, optimized_Amp, optimized_f_p0 = result1.x

# Convergence check
if result1.success:   
    print("Optimization converged successfully.")
else:
    print("Optimization did not converge.")

AIC = calculate_aic(result1.fun, 6)
AIC1 = calculate_aic(result.fun, 4)
print(f"AIC noise = {AIC1}, AIC signal = {AIC}")

# Printing optimized parameters
print(f"Optimized A1: {optimized_A1}")
print(f"Optimized A2: {optimized_A2}")
print(f"Optimized alpha1: {optimized_alpha1}")
print(f"Optimized alpha2: {optimized_alpha2}")
print(f"Optimized A: {optimized_A}")
print(f"Optimized P: {optimized_P}")
print(f"Optimized Amplitude: {optimized_Amp}")
print(f"Optimized Peak Frequency: {optimized_f_p0}")

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

# # Define the chi-squared function
# def chi_squared(params, frequencies, powerspectrum, mean_sample_data, standard_deviation, N_c):
#     A1, A2, alpha1, alpha2, A, P, Amp, f_peak = params
    
#     # Calculate the noise model and GW model
#     noise_model = lisa_noise_1(frequencies, A1, A2, alpha1, alpha2, A, P)
#     GW_model = powerspectrum.Omega_GW(frequencies, Amp, f_peak)

#     # Calculate the chi-squared value
#     residuals = (mean_sample_data - GW_model - noise_model) / standard_deviation
#     chi_2 = np.sum(residuals**2, axis=0)
    
#     return N_c * chi_2

# # Function for optimization
# def optimize_parameters(frequencies, powerspectrum, mean_sample_data, standard_deviation, N_c):
#     # Initial parameters
#     initial_params = [7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15, 1e-9, 1e-3] 

#     # Minimize the chi-squared function
#     result = minimize(chi_squared, initial_params, 
#                       args=(frequencies, powerspectrum, mean_sample_data, standard_deviation, N_c), 
#                       method='Powell', tol=1e-20)

#     # Extracting the optimized parameters
#     optimized_params = result.x
#     Chi_Squared_bf = result.fun

#     # Convergence check
#     success_message = "Optimization converged successfully." if result.success else "Optimization did not converge."
    
#     return optimized_params, Chi_Squared_bf, success_message


# # Call the optimization function
# optimized_params, Chi_Squared_bf, convergence_message = optimize_parameters(frequencies, powerspectrum, mean_sample_data, standard_deviation, N_c)

# # Print results
# print(f"Best fit parameters: {optimized_params}")
# print(f"Best fit Chi Squared value: {Chi_Squared_bf}")
# print(convergence_message)
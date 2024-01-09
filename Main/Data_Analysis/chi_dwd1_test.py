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

def chi_squared(params, frequencies, powerspectrum, mean_sample_data, standard_deviation):
    A1, A2, alpha1, alpha2, A, P, Amp, f_peak = params
    noise_model = lisa_noise_1(frequencies, A1, A2, alpha1, alpha2, A, P)
    GW_model = powerspectrum.Omega_GW(frequencies, Amp, f_peak)

    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - GW_model - noise_model) / standard_deviation) ** 2)
    print(chi_2)
    chi_2_value = len(frequencies) * np.sum(chi_2, axis=0)
    print("The chi-squared value is", chi_2_value)
    return chi_2_value

def optimize_parameters(frequencies, powerspectrum, N_c):
    # Generating the mock data, taking the mean and the standard deviation
    DATA = make_data_DWD_1(frequencies, N_c, powerspectrum)
    mean_sample_data = np.mean(DATA, axis=1)
    standard_deviation = np.std(DATA, axis=1)

    # Initial parameter guess
    initial_params = [7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15, 1e-9, 1e-3]

    # Minimize the chi-squared function
    result = minimize(chi_squared, initial_params, args=(frequencies, powerspectrum, mean_sample_data, standard_deviation), method='Powell', tol=1e-20)
    Chi_Squared_bf = result.fun
    print(f"Best fit Chi Squared value:", Chi_Squared_bf)

    # Extracting the optimized parameters
    optimized_A1, optimized_A2, optimized_alpha1, optimized_alpha2, optimized_A, optimized_P, optimized_Amp, optimized_f_p0 = result.x
    print(f"Optimized A1: {optimized_A1}")
    print(f"Optimized A2: {optimized_A2}")
    print(f"Optimized alpha1: {optimized_alpha1}")
    print(f"Optimized alpha2: {optimized_alpha2}")
    print(f"Optimized A: {optimized_A}")
    print(f"Optimized P: {optimized_P}")
    print(f"Optimized Amplitude: {optimized_Amp}")
    print(f"Optimized Peak Frequency: {optimized_f_p0}")

    # Convergence check
    if result.success:
        print("Optimization converged successfully.")
    else:
        print("Optimization did not converge.")

# Specifying frequency range and number of datapoints:
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
N_c = 94

# Create a PowerSpectrum object
powerspectrum = PowerSpectrum(0.6, 50, 180, 0.8)

# Perform optimization
optimize_parameters(frequencies, powerspectrum, N_c)
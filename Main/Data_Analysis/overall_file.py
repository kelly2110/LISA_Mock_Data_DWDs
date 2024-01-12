"""Encompassing the entire data structure into a single file.
Provides the output in various folders in order to ensure easy access to data analysis"""
# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from combined_data_gen import make_data_DWD_1, make_data_DWD_2, make_data_no_DWD
from all_chi import chi_squared_case_0_noise, chi_squared_case_0_signal
from noise import lisa_noise_1, lisa_noise_2
from SNR_calculation import calculate_snr_
from AIC import calculate_aic
import time
from ps_rewritten import PowerSpectrum
import numpy as np
import os
import inspect

# Specifying the range of PT parameters to create the powerspectrum
alpha_values = np.arange(0.1, 0.3, 0.1)
vw_values = np.arange(0.5, 0.7, 0.1)
beta_over_h_values = np.arange(50, 150, 10)

# Specifying the frequency range to be used
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
N_c = 50 # Number of data points used for analysis

# Creating folders to store data in
folder_name0 = "PS_0bjects"
os.makedirs(folder_name0, exist_ok=True)

# Create multiple powerspectrum objects
power_spectrum_objects = np.empty((len(alpha_values), len(vw_values)), dtype=object) # Specifying the size of the array

# Object Creation for each combination of alpha and vw
for i, alpha in enumerate(alpha_values):
    for j, vw in enumerate(vw_values):
        PS = PowerSpectrum(alpha, 50, 180, vw)
        GW = PS.Omega_GW(frequencies, PS.Amp, PS.fp_0())
        power_spectrum_objects[i, j] = GW
        data = np.column_stack((frequencies, GW))

# Loop over power_spectrum_objects
for i, alpha in enumerate(alpha_values):
    for j, vw in enumerate(vw_values):
        power_spectrum_object = power_spectrum_objects[i, j]

        # Creating the mock data that goes into the chi squared
        DWD0 = make_data_no_DWD(frequencies, N_c, power_spectrum_object)
        mean = np.mean(DWD0, axis=1)
        std_dev = np.std(DWD0, axis=1)

        # Calculating the chi squared
                # Define initial parameters for optimization
        initial_params = [3, 15]

        # Define the chi-squared function with only optimization parameters
        chi = lambda params: chi_squared_case_0_noise(params, frequencies, N_c, mean, std_dev)
        # Minimize the chi-squared function
        result = minimize(chi, initial_params, args=(frequencies, mean, std_dev), method='Powell')
        A_opt, P_opt = result.x
        Chi_Squared_bf = result.fun
        A_opt, P_opt = result.x
        Chi_Squared_bf = result.fun

        # Print the results
        print(f"For alpha={alpha}, vw={vw}, Best fit A={A_opt}, Best fit P={P_opt}, Best fit Chi Squared value:", Chi_Squared_bf)

        # Print the comparison of optimized and initial parameters
        print(f"Comparison - Initial A={initial_params[0]}, Initial P={initial_params[1]}, Optimal A={A_opt}, Optimal P={P_opt}")


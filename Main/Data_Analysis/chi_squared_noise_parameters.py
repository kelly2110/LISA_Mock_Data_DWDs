# Testing X^2 Minimization for noise + signal, with parameters A & P.
# Using the noise model of Giese et al.

import numpy as np
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps_rewritten import PowerSpectrum
from combined_data_gen import make_data_no_DWD
import time

# Define the chi-squared function 
def chi_squared_case_0_noise(params, frequencies, mean_sample_data, GW_model, standard_deviation):
    chi_2_value = []
    A, P = params
    noise_model = Omega_N(frequencies, A, P)    

    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - GW_model - noise_model) / standard_deviation)**2)
    print("chi squared:",chi_2[0:2])
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    print(" The chi squared value is", chi_2_value)
    return chi_2_value

# Generate sample data with true parameters 
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
N_c = 60
Omega_Noise = Omega_N(frequencies, 3,15)
P1 = PowerSpectrum(0.6, 50, 180, 0.8)
GW_model = P1.Omega_GW(frequencies, P1.Amp, P1.fp_0())
DATA = make_data_no_DWD(frequencies, N_c, P1)
mean_sample_data = np.mean(DATA, axis=1)
standard_deviation = np.std(DATA, axis=1)

# Initial guess for parameters
initial_params = [3, 15] 

# Minimize the chi-squared function
result = minimize(chi_squared_case_0_noise, initial_params, args=(frequencies, mean_sample_data, GW_model, standard_deviation), method='Powell')
Chi_Squared_bf = result.fun
print(f"Best fit Chi Squared value:", Chi_Squared_bf)

optimized_A, optimized_P = result.x # Extract the optimized parameters
print(f"Optimized A: {optimized_A}")
print(f"Optimized P: {optimized_P}")

# Checking for convergence
if result.success:   
    print("Optimization converged successfully.")
else:
    print("Optimization did not converge.")


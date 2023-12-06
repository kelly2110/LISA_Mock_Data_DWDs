# Testing X^2 Minimization for noise only, with parameters A & P.
# Using the noise model of Giese et al.

import numpy as np
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps import PowerSpectrum
from data_gen_noise import calculate_N

# Calculating X^2 values

# Define the chi-squared function for LISA noise model
def chi_squared(params, frequencies, mean_sample_data, standard_deviation):
    chi_2 = np.zeros((len(frequencies)))
    A, P = params
    noise_model = Omega_N(frequencies, A, P)
    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - noise_model) / standard_deviation)**2)
    chi_2_value = N_c*np.sum(chi_2)
    print(" The chi squared value is", chi_2_value)
    return chi_2_value

# Generate sample data with true parameters 
f_low = np.arange(0.00003, 0.001, 0.000001)
f_high = np.arange(0.001, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_high))
N_c = 50
A = 3
P = 15
Omega_Noise = calculate_N(frequencies, N_c, A, P)
mean_sample_data = np.mean(Omega_Noise, axis=1)
standard_deviation = np.std(Omega_Noise, axis=1)

# Initial guess for parameters
initial_params = [6, 25] 

# Minimize the chi-squared function
result = minimize(chi_squared, initial_params, args=(frequencies, mean_sample_data, standard_deviation), method='Powell')
Chi_Squared_bf = result.fun
print(f"Best fit Chi Squared value:", Chi_Squared_bf)
optimized_A, optimized_P = result.x # Extract the optimized parameters
print(f"Optimized A: {optimized_A}")
print(f"Optimized P: {optimized_P}")

if result.success:   
    print("Optimization converged successfully.")
# Checking for convergence
else:
    print("Optimization did not converge.")


def calculate_aic(chi, k):
    AIC = chi + 2*k
    print("The AIC value is:", AIC)
    return AIC

calculate_aic(Chi_Squared_bf, 2)


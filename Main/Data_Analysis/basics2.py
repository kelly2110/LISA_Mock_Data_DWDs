# Testing X^2 Minimization for noise only, with parameters A & P.
# Using the noise model of Giese et al.

import numpy as np
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps import PowerSpectrum
from data_gen_noise import calculate_N

""" # Defining Sample Data
Omega_Noise = calculate_N(f, N_c)
mean_Omega_Noise = np.mean(Omega_Noise, axis=0)
std_dev_Omega_Noise = np.std(Omega_Noise, axis=0)
print("Shape of Mean data array:", mean_Omega_Noise)
print("Shape of Standard deviation data array:", std_dev_Omega_Noise) """

# Calculating X^2 values

# Define the chi-squared function for LISA noise model
def chi_squared(params, frequencies):
    chi_2_value = []
    A, P = params
    noise_model = Omega_N(frequencies, A, P)
    Omega_Noise = calculate_N(frequencies, N_c, A, P)
    mean_sample_data = np.mean(Omega_Noise, axis=0)
    standard_deviation = np.std(Omega_Noise, axis=0)
    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - noise_model) / standard_deviation)**2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    print(" The chi squared value is", chi_2_value)
    return chi_2_value

# Generate sample data with true parameters 
frequencies = np.linspace(3e-5, 0.5, 5000)
N_c = 50

# Initial guess for parameters
initial_params = [0.1, 0.4] 

# Minimize the chi-squared function
result = minimize(chi_squared, initial_params, args=(frequencies), method='Powell')

# Extract the optimized parameters
optimized_A, optimized_P = result.x

print(f"Optimized A: {optimized_A}")
print(f"Optimized P: {optimized_P}")


# Checking for convergence
if result.success:
    print("Optimization converged successfully.")
else:
    print("Optimization did not converge.")
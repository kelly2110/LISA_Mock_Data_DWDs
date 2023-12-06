import numpy as np
from scipy.optimize import minimize
from giese_lisa_sens import Omega_N
from ps_rewritten import PowerSpectrum
from data_gen_noise import calculate_N
from combined_noise_signal_data import make_data
from kinetic_energy_fraction import KineticEnergyFraction
from numpy import pi, sqrt

# Define the chi-squared function for noise + signal
def chi_squared_combined(params, frequencies, mean_sample_data, standard_deviation, N_c):
    A, P, Amp, f_peak = params
    
    # Calculate noise model
    noise_model = Omega_N(frequencies, A, P)
    
    # Calculate signal model
    PS = PowerSpectrum(Amp, P, 180, 0.8)
    signal_model = PS.Omega_GW(frequencies)
    
    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - signal_model - noise_model) / standard_deviation)**2)
    chi_2_value = N_c * np.sum(chi_2, axis=0)
    
    print("The chi squared value is", chi_2_value)
    return chi_2_value

# Generate sample data with true parameters 
f_low = np.arange(0.00003, 0.001, 0.000001)
f_high = np.arange(0.001, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_high))
N_c = 50
Omega_Noise = calculate_N(frequencies, N_c, 3, 15)
P1 = PowerSpectrum(0.6, 50, 180, 0.8)
GW_model = P1.Omega_GW(frequencies)
DATA = make_data(frequencies, N_c, 3, 15, P1)
mean_sample_data = np.mean(DATA, axis=0)
standard_deviation = np.std(DATA, axis=0)

# Initial guess for parameters
initial_params = [3, 15, 1e-15, 0.01]  # A, P, Amp, f_peak

# Minimize the chi-squared function
result = minimize(chi_squared_combined, initial_params, 
                  args=(frequencies, mean_sample_data, standard_deviation, N_c),
                  method='Powell')

Chi_Squared_bf = result.fun
print(f"Best fit Chi Squared value:", Chi_Squared_bf)

optimized_A, optimized_P, optimized_Amp, optimized_f_peak = result.x  # Extract the optimized parameters
print(f"Optimized A: {optimized_A}")
print(f"Optimized P: {optimized_P}")
print(f"Optimized Amp: {optimized_Amp}")
print(f"Optimized f_peak: {optimized_f_peak}")

if result.success:   
    print("Optimization converged successfully.")
else:
    print("Optimization did not converge.")

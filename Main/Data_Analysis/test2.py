import numpy as np
from scipy.optimize import minimize

# Constants
h = 0.678
H_0 = (100 * h) / (3.09e19)  # km sec^-1 Mpc^-1 --> 1/s --> Hz
c = 3.0e8  # m/s
L = 2.5e9  # m
f_s = c / (2 * np.pi * L)

# Noise Parameters, P = 15, A = 3
A = 3
P = 15

frequencies = np.logspace(-4, 0, 100)

# Noise model
from giese_lisa_sens import S_n, Omega_N
Noise_obs = Omega_N(frequencies, A, P)
print(Noise_obs)
Noise_obs_mean = np.mean(Noise_obs) #Mean
print(Noise_obs_mean) 
Noise_obs_st_dev = np.std(Noise_obs) # Should be standard deviation
print(Noise_obs_st_dev)

# Defining chi-squared function, not including mean and st. dev. yet
def chi_squared(params, f, data_mean, data, data_std):
    model_data = Omega_N(f, params[0], params[1])  # A and P are parameters
    chi_sq = np.sum(((data_mean - data - model_data) / data_std) ** 2)
    return chi_sq

# Initial guess for parameters (A, P)
initial_params = [2, 30]

# Minimize the chi-squared function for best-fit parameters
Chi_Squared_Min = minimize(chi_squared, initial_params, args=(frequencies, Noise_obs, Noise_obs_mean, Noise_obs_st_dev))

# Extract the best-fit parameters
best_params = Chi_Squared_Min.x
print("Best-fitting parameters (A, P):", best_params)
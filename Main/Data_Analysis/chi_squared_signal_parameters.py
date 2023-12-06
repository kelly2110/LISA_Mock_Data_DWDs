import numpy as np
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps_rewritten import PowerSpectrum
from data_gen_noise import calculate_N
from correct_data_gen_signal_2 import calculate_S
from combined_data_gen import make_data
import time

# Define the chi-squared function 
start_time = time.time()
def chi_squared(params, frequencies, powerspectrum, noise_model, mean_sample_data, standard_deviation):
    chi_2 = []
    Amp, f_peak = params
    GW_model = powerspectrum.Omega_GW(frequencies, Amp, f_peak)
    
    # Calculate the chi-squared value
    chi_2 = (((mean_sample_data - GW_model - noise_model) / standard_deviation)**2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    print(" The chi squared value is", chi_2_value)
    return chi_2_value

# Generate sample data with true parameters 
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
N_c = 50
noise_model = Omega_N(frequencies, 3, 15)
powerspectrum = PowerSpectrum(0.6, 50, 180, 0.8)
DATA = make_data(frequencies, N_c, 3, 15, powerspectrum)
mean_sample_data = np.mean(DATA, axis=1)
standard_deviation = np.std(DATA, axis=1)

# Initial guess for parameters, Amp = 1.386e-9, peak frequency = 1.005e-3
initial_params = [1e-9, 1e-3] 

# Minimize the chi-squared function
result = minimize(chi_squared, initial_params, args=(frequencies, powerspectrum, noise_model, mean_sample_data, standard_deviation), method='Powell', tol=1e-11)
Chi_Squared_bf = result.fun
print(f"Best fit Chi Squared value:", Chi_Squared_bf)

optimized_Amp, optimized_f_p0 = result.x # Extract the optimized parameters
print(f"Optimized Amplitude: {optimized_Amp}")
print(f"Optimized Peak Frequency: {optimized_f_p0}")

if result.success:   
    print("Optimization converged successfully.")
# Checking for convergence
else:
    print("Optimization did not converge.")

# Two tails, amplitude A and peak frequency f_p0, so k = 4
def calculate_aic(chi):
    k = 4
    AIC = chi + 2*k
    print("The AIC value is:", AIC)
    return AIC

calculate_aic(Chi_Squared_bf)
end_time = time.time()
run_time = end_time - start_time
print(f"The run time is {run_time} seconds")
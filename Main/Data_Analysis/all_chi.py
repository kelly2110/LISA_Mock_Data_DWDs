import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps_rewritten import PowerSpectrum
from combined_data_gen import make_data_no_DWD
from noise import lisa_noise_1, lisa_noise_2

# Chi squared functions when no DWD background is present

def chi_squared_case_0_noise(params, frequencies, N_c, mean, std_dev):
    chi_2 = []
    A, P = params
    noise_model = Omega_N(frequencies, A, P)
    chi_2 = (((mean - noise_model) / std_dev)**2)
    chi_2_value = N_c*np.sum(chi_2)
    return chi_2_value


def chi_squared_case_0_signal(params, frequencies, N_c, powerspectrum, mean, std_dev):
    chi_2 = []
    A, P, Amp, f_peak = params
    noise_model = Omega_N(frequencies, A, P)
    GW_model = powerspectrum.Omega_GW(frequencies, Amp, f_peak)
    chi_2 = (((mean - GW_model - noise_model) / std_dev)**2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    return chi_2_value


# Chi squared functions for the DWD foreground 1
def chi_squared_case_1_noise(params, frequencies, N_c,  mean, std_dev):
    chi_2 = []
    A1, A2, alpha1, alpha2, A, P = params
    noise_model = lisa_noise_1(frequencies, A1, A2, alpha1, alpha2, A, P)
    chi_2 = (((mean - noise_model) / std_dev)**2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    return chi_2_value

def chi_squared_case_1_signal(params, frequencies, N_c, powerspectrum, mean_sample_data, standard_deviation):
    chi_2 = []
    A1, A2, alpha1, alpha2, A, P, Amp, f_peak  = params
    noise_model = lisa_noise_1(frequencies, A1, A2, alpha1, alpha2, A, P)
    GW_model = powerspectrum.Omega_GW(frequencies, Amp, f_peak)
    chi_2 = (((mean_sample_data - GW_model - noise_model) / standard_deviation)**2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    return chi_2_value

# Chi squared functions for the DWD foreground 2
def chi_squared_case_2_noise(params, frequencies, N_c,  mean, std_dev):
    chi_2 = []
    alpha, beta, gamma, k, A, P = params
    noise_model = lisa_noise_2(frequencies, alpha, beta, gamma, k, A, P)
    chi_2 = (((mean - noise_model) / std_dev)**2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    return chi_2_value

def chi_squared_case_2_signal(params, frequencies, N_c, powerspectrum, mean_sample_data, standard_deviation):
    chi_2 = []
    alpha, beta, gamma, k, A, P , Amp, f_peak = params
    noise_model = lisa_noise_2(frequencies, alpha, beta, gamma, k, A, P)
    GW_model = powerspectrum.Omega_GW(frequencies, Amp, f_peak)
    chi_2 = (((mean_sample_data - GW_model - noise_model) / standard_deviation)**2)
    chi_2_value = N_c*np.sum(chi_2, axis=0)
    return chi_2_value

def minimize_chi_squared(initial_params, chi_squared_function, *args):
    result = minimize(chi_squared_function, initial_params, args=args, method='Powell')
    return result.x, result.fun
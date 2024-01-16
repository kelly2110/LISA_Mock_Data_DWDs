"""Encompassing the entire data structure into a single file.
Provides the output in various folders in order to ensure easy access to data analysis"""
import numpy as np
import matplotlib.pyplot as plt
from giese_lisa_sens import Omega_N
from combined_data_gen import make_data_DWD_1
from all_chi import chi_squared_case_1_noise, chi_squared_case_1_signal, minimize_chi_squared
from SNR_calculation import calculate_snr_
from AIC import calculate_aic
from ps_rewritten import PowerSpectrum
import os

# Specifying the range of PT parameters to create the powerspectrum
temperature_values = np.arange(180, 190, 10)
alpha_values = np.arange(0.001,0.01,0.001).round(3)
vw_values = np.arange(0.3,0.9,0.1).round(2)
beta_over_h = 100

# Specifying the frequency range to be used
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
N_c = 50 # Number of data points used for analysis

# Creating folders to store data in
main_folder = "DWD_1"
os.makedirs(main_folder, exist_ok=True)
data_points_folder = os.path.join(main_folder, f"{N_c}_data_points")
os.makedirs(data_points_folder, exist_ok=True)
gw_data_folder = os.path.join(data_points_folder, "GW_data")
chi_squared_folder = os.path.join(data_points_folder, "chi_squared")
os.makedirs(gw_data_folder, exist_ok=True)
os.makedirs(chi_squared_folder, exist_ok=True)

# Create empty arrays to store data
index = 0 # Index to track the current row in the arrays
chi_squared_data = np.empty((len(alpha_values) * len(vw_values) * len(temperature_values), 5))
aic_data = np.empty((len(alpha_values) * len(vw_values) * len(temperature_values), 5))
snr_data = np.empty((len(alpha_values) * len(vw_values) * len(temperature_values), 5))
power_spectrum_objects = np.empty((len(alpha_values), len(vw_values)), dtype=object)
parameters_data = np.empty((len(alpha_values) * len(vw_values) * len(temperature_values), 17))

for temp in temperature_values:
    for i, alpha in enumerate(alpha_values):
        for j, vw in enumerate(vw_values):
                PS = PowerSpectrum(alpha, beta_over_h, temp, vw)
                GW_model = PS.Omega_GW(frequencies, PS.Amp, PS.fp_0())
                Noise = Omega_N(frequencies, 3, 15)
                power_spectrum_objects[i, j] = PS
                power_spectrum_object = power_spectrum_objects[i, j]

                # Creating the mock data that goes into the chi squared
                DWD1 = make_data_DWD_1(frequencies, N_c, power_spectrum_object)
                mean_data = np.mean(DWD1, axis=1)
                std_dev_data = np.std(DWD1, axis=1)

                # Save mean and standard deviation to a single CSV file
                filename_m_std = f"combined_data_alpha_{alpha}_vw_{vw}_temp_{temp}.csv"
                filepath_combined = os.path.join(gw_data_folder, filename_m_std)
                header_combined = ["Frequency", "GW_model data",  "Mean", "Std Dev"]
                data_to_save_combined = np.column_stack((frequencies, GW_model, mean_data, std_dev_data))
                np.savetxt(filepath_combined, data_to_save_combined, delimiter=',', header=','.join(header_combined), comments='')

                # Calculating and minimizing the chi squared
                initial_params_0 = [7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15]
                initial_params_1 = [7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15, power_spectrum_object.Amp, power_spectrum_object.fp_0()] 

                chi_noise = lambda params, *args: chi_squared_case_1_noise(params, *args)
                chi_signal = lambda params, *args: chi_squared_case_1_signal(params, *args)

                result_0 = minimize_chi_squared(initial_params_0, chi_noise, frequencies, N_c, mean_data, std_dev_data)
                result_1 = minimize_chi_squared(initial_params_1, chi_signal, frequencies, N_c, power_spectrum_object, mean_data, std_dev_data)

                # Unpack the results
                A1_opt_0, A2_opt_0, alpha1_opt_0, alpha2_opt_0, A_opt_0, P_opt_0 = result_0[0][:6]
                Chi_Squared_bf_n = result_0[1]

                A1_opt, A2_opt, alpha1_opt, alpha2_opt, A_opt, P_opt, Amp_opt, fp_opt = result_1[0][:8]
                Chi_Squared_bf_s = result_1[1]

                aic_n = calculate_aic(Chi_Squared_bf_n, 6)
                aic_s = calculate_aic(Chi_Squared_bf_s, 8)
                print(f"For alpha={alpha}, vw={vw},  temperature={temp}, the noise only AIC value is: {aic_n} \n and the signal AIC value is: {aic_s}")
                SNR_original = calculate_snr_(GW_model, Noise, frequencies)
                SNR_reconstructed = calculate_snr_(power_spectrum_object.Omega_GW(frequencies, Amp_opt, fp_opt), Noise, frequencies)

                chi_squared_data[index] = [alpha, vw, temp, Chi_Squared_bf_n, Chi_Squared_bf_s]
                aic_data[index] = [alpha, vw, temp, aic_n, aic_s]
                snr_data[index] = [alpha, vw, temp, SNR_original, SNR_reconstructed]
                parameters_data[index] = [alpha, vw, temp, A_opt_0, A_opt, P_opt_0, P_opt, A1_opt_0, A1_opt, A2_opt_0, A2_opt, alpha1_opt_0, alpha1_opt, alpha2_opt_0, alpha2_opt, Amp_opt, fp_opt]
                index += 1

filename_chi_squared = "results_chi_squared.csv"
filepath_chi_squared = os.path.join(chi_squared_folder, filename_chi_squared)
header_chi_squared = ["Alpha", "vw", "Temperature", "Chi_Squared_Noise", "Chi_Squared_Signal"]

filename_aic = "results_aic.csv"
filepath_aic = os.path.join(chi_squared_folder, filename_aic)
header_aic = ["Alpha", "vw", "Temperature", "AIC_Noise", "AIC_Signal"]

filename_snr = "results_snr.csv"
filepath_snr = os.path.join(chi_squared_folder, filename_snr)
header_snr = ["Alpha", "vw", "Temperature", "SNR_original", "SNR_reconstructed"]

filename_parameters = "results_parameters.csv"
filepath_parameters = os.path.join(chi_squared_folder, filename_parameters)
header_parameters = ["Alpha", "vw", "Temperature", "A_opt_noise", "A_opt_n+s", "P_opt_noise", "P_opt_n+s", "A1_opt_noise", "A1_opt_n+s", "A2_opt_noise", "A2_opt_n+s", "alpha1_opt_noise", "alpha1_opt_n+s", "alpha2_opt_noise", "alpha2_opt_n+s", "Amp_opt", "fp_opt_signal"]

np.savetxt(filepath_chi_squared, chi_squared_data, delimiter=',', header=','.join(header_chi_squared), comments='')
np.savetxt(filepath_aic, aic_data, delimiter=',', header=','.join(header_aic), comments='')
np.savetxt(filepath_snr, snr_data, delimiter=',', header=','.join(header_snr), comments='')
np.savetxt(filepath_parameters, parameters_data, delimiter=',', header=','.join(header_parameters), comments='')
print("Process Completed!")
"""Encompassing the entire data structure into a single file.
Provides the output in various folders in order to ensure easy access to data analysis"""
import numpy as np
from giese_lisa_sens import Omega_N
from combined_data_gen import make_data_DWD_2
import matplotlib.pyplot as plt
from noise import lisa_noise_2
from all_chi import chi_squared_case_2_noise, chi_squared_case_2_signal, minimize_chi_squared
from SNR_calculation import calculate_snr_
from AIC import calculate_aic
from ps_rewritten import PowerSpectrum
import os

# Specifying the range of PT parameters to create the powerspectrum
temperature_values = np.arange(180, 190, 10)
alpha_values = np.arange(0.1, 0.4, 0.05).round(3)
vw_values = np.arange(0.35,0.4,0.2).round(2)
beta_over_h = 100

# Specifying the frequency range to be used
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
N_c = 95 # Number of data points used for analysis

# Creating folders to store data in
main_folder = "DWD_2"
os.makedirs(main_folder, exist_ok=True)
data_points_folder = os.path.join(main_folder, f"{N_c}_data_points")
os.makedirs(data_points_folder, exist_ok=True)
gw_data_folder = os.path.join(data_points_folder, "GW_data")
os.makedirs(gw_data_folder, exist_ok=True)

# Create empty arrays to store data
chi_squared_data = np.empty((len(alpha_values) * len(vw_values) * len(temperature_values), 5))
aic_data = np.empty((len(alpha_values) * len(vw_values) * len(temperature_values), 5))
snr_data = np.empty((len(alpha_values) * len(vw_values) * len(temperature_values), 5))
power_spectrum_objects = np.empty((len(alpha_values), len(vw_values)), dtype=object)
parameters_data = np.empty((len(alpha_values) * len(vw_values) * len(temperature_values), 9))

for temp in temperature_values:
    for i, alpha in enumerate(alpha_values):
        for j, vw in enumerate(vw_values):
                PS = PowerSpectrum(alpha, beta_over_h, temp, vw)
                GW_model = PS.Omega_GW(frequencies, PS.Amp, PS.fp_0())
                Noise = lisa_noise_2(frequencies, 0.138, -221, 521, 1680, 3, 15)
                power_spectrum_objects[i, j] = PS
                power_spectrum_object = power_spectrum_objects[i, j]

                # Creating the mock data that goes into the chi squared
                DWD2 = make_data_DWD_2(frequencies, N_c, power_spectrum_object)
                mean_data = np.mean(DWD2, axis=1)
                std_dev_data = np.std(DWD2, axis=1)

                # Save mean and standard deviation to a single CSV file
                filename_m_std = f"combined_data_alpha_{alpha}_vw_{vw}_temp_{temp}.csv"
                filepath_combined = os.path.join(gw_data_folder, filename_m_std)
                header_combined = ["Frequency", "GW_model data",  "Mean", "Std Dev"]
                data_to_save_combined = np.column_stack((frequencies, GW_model, mean_data, std_dev_data))
                np.savetxt(filepath_combined, data_to_save_combined, delimiter=',', header=','.join(header_combined), comments='')

                # Calculating and minimizing the chi squared
                initial_params_0 = [0.138, -221, 521, 1680, 3, 15]
                initial_params_1 = [0.138, -221, 521, 1680, 3, 15, power_spectrum_object.Amp, power_spectrum_object.fp_0()] 

                chi_noise = lambda params, *args: chi_squared_case_2_noise(params, *args)
                chi_signal = lambda params, *args: chi_squared_case_2_signal(params, *args)

                result_0 = minimize_chi_squared(initial_params_0, chi_noise, frequencies, N_c, mean_data, std_dev_data)
                result_1 = minimize_chi_squared(initial_params_1, chi_signal, frequencies, N_c, power_spectrum_object, mean_data, std_dev_data)

                # Unpack the results
                alpha_opt_0, beta_opt_0, kappa_opt_0, gamma_opt_0, A_opt_0, P_opt_0 = result_0[0][:6]
                chi_squared_bf_n = result_0[1]

                alpha_opt, beta_opt, kappa_opt, gamma_opt, A_opt, P_opt, Amp_opt, fp_opt = result_1[0][:8]
                chi_squared_bf_s = result_1[1]

                aic_n = calculate_aic(chi_squared_bf_n, 6)
                aic_s = calculate_aic(chi_squared_bf_s, 8)
                print(f"For alpha={alpha}, vw={vw},  temperature={temp}, the noise only AIC value is: {aic_n} \n and the signal AIC value is: {aic_s}")
                SNR_original = calculate_snr_(GW_model, Noise, frequencies)
                SNR_reconstructed = calculate_snr_(power_spectrum_object.Omega_GW(frequencies, Amp_opt, fp_opt), Noise, frequencies)

                               # Adjust the folder name to include alpha, vw, and temperature values
                folder_name = f"alpha_{alpha}_vw_{vw}_temp_{temp}"
                datapoints_folder_alpha_vw_temp = os.path.join(data_points_folder, folder_name)
                os.makedirs(datapoints_folder_alpha_vw_temp, exist_ok=True)

                # Adjust the filenames to include alpha, vw, and temperature values
                filename_chi_squared = f"results_chi_squared.csv"
                filepath_chi_squared = os.path.join(datapoints_folder_alpha_vw_temp, filename_chi_squared)

                filename_aic = f"results_aic.csv"
                filepath_aic = os.path.join(datapoints_folder_alpha_vw_temp, filename_aic)

                filename_snr = f"results_snr.csv"
                filepath_snr = os.path.join(datapoints_folder_alpha_vw_temp, filename_snr)

                filename_parameters = f"results_parameters.csv"
                filepath_parameters = os.path.join(datapoints_folder_alpha_vw_temp, filename_parameters)

                data_chi_squared = np.array([[chi_squared_bf_n, chi_squared_bf_s]])
                data_aic = np.array([[aic_n, aic_s]])
                data_snr = np.array([[SNR_original, SNR_reconstructed]])
                data_parameters = np.array([[A_opt_0, A_opt, P_opt_0, P_opt, alpha_opt_0, alpha_opt, beta_opt_0, beta_opt, kappa_opt_0, kappa_opt, gamma_opt_0, gamma_opt, power_spectrum_object.Amp, Amp_opt, power_spectrum_object.fp_0(), fp_opt]])
                GW_recon = PS.Omega_GW(frequencies, Amp_opt, fp_opt )
               
                header_chi_squared = ["chi_squared_noise", "chi_squared_signal"]
                header_aic = [ "AIC_noise", "AIC_signal"]
                header_snr = [ "snr_original", "snr_reconstructed"]
                header_parameters = ["A_opt_noise", "A_opt_n+s", "P_opt_noise", "P_opt_n+s", "alpha_opt_noise", "alpha_opt_n+s", "beta_opt_noise", "beta_opt_n+s", "kappa_opt_noise", "kappa_opt_n+s", "gamma_opt_noise", "gamma_opt_n+s", "initial_Amp", "Amp_opt", "initial_peak", "fp_opt_signal"]

                with open(filepath_chi_squared, 'a') as file:
                            # Append the data for the current iteration to the file
                            np.savetxt(file, data_chi_squared, delimiter=',', header=','.join(header_chi_squared), comments='')

                with open(filepath_aic, 'a') as file:
                            # Append the data for the current iteration to the file
                            np.savetxt(file, data_aic, delimiter=',', header=','.join(header_aic), comments='')

                with open(filepath_snr, 'a') as file:
                            # Append the data for the current iteration to the file
                            np.savetxt(file, data_snr, delimiter=',', header=','.join(header_snr), comments='')

                with open(filepath_parameters, 'a') as file:
                            # Append the data for the current iteration to the file
                            np.savetxt(file, data_parameters, delimiter=',', header=','.join(header_parameters), comments='')
                plt.figure(figsize=(8, 6))  # Adjust the figsize as needed
                plt.loglog(frequencies, mean_data, label='Generated Mock Data', color='b')                   
                plt.loglog(frequencies, GW_model, label='Original Signal', color='c')
                plt.loglog(frequencies, GW_recon, label='Reconstructed Signal', color='g')
                plt.loglog(frequencies, Omega_N(frequencies, 3, 15), label='LISA Sensitivity', color='k')
                plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
                plt.ylabel(r'$h^{2}\Omega(f)$')
                plt.title(r'Original and reconstructed gravitational wave signal')

                # Display SNR and AIC values on the right of the plot
                snr_text = f'SNR: {SNR_original:.4f}'
                aic_text = f'AIC: {aic_s:.4f}'

                plt.text(0.02, 0.06, snr_text, transform=plt.gca().transAxes, fontsize=10, color='k')
                plt.text(0.02, 0.02, aic_text, transform=plt.gca().transAxes, fontsize=10, color='k')


                # Save the figure with annotations
                plot_filename = f'Gravitational_wave_signal_alpha_{alpha}_vw_{vw}_temp_{temp}_with_annotations.png'
                plt.legend()
                plt.savefig(plot_filename)


                # Show the figure with annotations
                plt.show()
print("Process Completed!")
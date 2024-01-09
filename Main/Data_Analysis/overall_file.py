"""Encompassing the entire data structure into a single file.
Provides the output in various folders in order to ensure easy access to data analysis"""

# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps_rewritten import PowerSpectrum
from combined_data_gen import make_data_DWD_1, make_data_DWD_2, make_data_no_DWD
from noise import lisa_noise_1, lisa_noise_2
from SNR_calculation import calculate_snr_
from AIC import calculate_aic
import time
from ps_rewritten import PowerSpectrum
import numpy as np
import os
import inspect

# Specifying the range of PT parameters to create the powerspectrum
alpha_values = np.arange(0.1, 0.9, 0.1)
vw_values = np.arange(0.1, 1.0, 0.1)
beta_over_h_values = np.arange(50, 150, 10)

# Specifying the frequency range to be used
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))


# Creating folders to store data in
folder_name0 = "GW_data"
folder_name1 = "mean_values"
folder_name2 = "std_dev_values"
folder_name3 = "chi_squared_outcomes"
folder_name4 = "AIC_outcomes"

os.makedirs(folder_name0, exist_ok=True)

# Create multiple powerspectrum objects, to be plugged into the chi square analysis
    # Create a new folder to save the results
folder_name = "Mock_Data_Results"
os.makedirs(folder_name, exist_ok=True)

for alpha in alpha_values:
        for vw in vw_values:
            # Object Creation for each combination of alpha and vw
            PS = PowerSpectrum(alpha, 50, 180, vw)
            GW_model = PS.Omega_GW(frequencies, PS.Amp, PS.fp_0())
            DWD0 = make_data_no_DWD(frequencies, 94, PS)
            DWD1 = make_data_DWD_1(frequencies, 94, PS)
            DWD2 = make_data_DWD_2(frequencies, 94, PS)
            Sensitivity = Omega_N(frequencies, 3, 15)

 # Calculate mean and standard deviation
            mean_std_dev_DWD0 = np.column_stack((np.mean(DWD0, axis=1), np.std(DWD0, axis=1)))
            mean_std_dev_DWD1 = np.column_stack((np.mean(DWD1, axis=1), np.std(DWD1, axis=1)))
            mean_std_dev_DWD2 = np.column_stack((np.mean(DWD2, axis=1), np.std(DWD2, axis=1)))

            # Save means and standard deviations of the generated mock data
            file_name_prefix = f'MockData_alpha_{alpha}_vw_{vw}'
            np.savetxt(os.path.join(folder_name, f'{file_name_prefix}_Mean_Std_DATA1.csv'), mean_std_dev_DWD0, delimiter=',', header='Mean, Std', comments='')
            np.savetxt(os.path.join(folder_name, f'{file_name_prefix}_Mean_Std_DATA2.csv'), mean_std_dev_DWD1, delimiter=',', header='Mean, Std', comments='')
            np.savetxt(os.path.join(folder_name, f'{file_name_prefix}_Mean_Std_DATA3.csv'), mean_std_dev_DWD2, delimiter=',', header='Mean, Std', comments='')

            print(f"Mock data for alpha = {alpha}, vw = {vw} saved.")

print("Process completed.")



# calculate the three chi squared functions for all powerspectrums


# Calculate AIC values for each case, noise and noise+signal
def auto_AIC(*args, **kwargs):
    # Get the number of parameters of the function
    num_params = len(inspect.signature().parameters)

    # Check if the required number of parameters for AIC is provided
    if len(args) < num_params:
        raise ValueError(f"Not enough parameters provided for calculate_one. Expected {num_params}, got {len(args)}.")

    # Extract the first 'num_params' parameters for calculate_one
    calculate_one_args = args[:num_params]

    # Call calculate_one with the extracted parameters
    result_calculate_one = calculate_one(*calculate_one_args)

    # Call AIC with the result of calculate_one and the fixed parameter 'y'
    result_AIC = AIC(result_calculate_one, y=3)

    return result_AIC


# save results for each chi squared in a separate file/folder
""" for aic in aic_values
file_name = f'Omega_GW_alpha_{alpha}_vw_{vw}.csv'
            file_path = os.path.join(folder_name4, file_name)
            np.savetxt(file_path, data, delimiter=',', header='Frequency, AIC values', comments='')
            print(f"For alpha = {alpha}, vw = {vw}, GW signal saved in {file_path}.") """


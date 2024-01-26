import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ps_rewritten import PowerSpectrum  # Replace with the actual import statement for your Powerspectrum class

# Main folder containing subfolders (e.g., DWD_2)
main_folder = 'DWD_0'
# Subfolder for 95_data_points
data_points_subfolder = '95_data_points'
# Constant values
constant_vw = 0.76
constant_temperature = 180
# Alpha values to plot
alpha_values_to_plot = [0.4]  # Use a list for multiple alpha values

f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))

# Initialize lists to store signals for each alpha
original_signals = []
reconstructed_signals = []

# Loop over all subfolders in the main folder
for root, dirs, files in os.walk(main_folder):
    for subfolder in dirs:
        # Check if the subfolder name represents the format {n_c}_data_points
        if subfolder == data_points_subfolder:
            data_points_folder_path = os.path.join(root, subfolder)

            # Loop over subfolders within 95_data_points
            for subfolder_95 in os.listdir(data_points_folder_path):
                subfolder_path = os.path.join(data_points_folder_path, subfolder_95)

                # Check if it's a directory, follows the naming convention, and temperature matches
                if os.path.isdir(subfolder_path) and subfolder_95.startswith('alpha_') and subfolder_95.endswith(f'_temp_{constant_temperature}'):
                    alpha_value = float(subfolder_95.split('_')[1])
                    vw_value = float(subfolder_95.split('_')[3])

                    # Check if alpha is in the list of alpha values to plot
                    if alpha_value in alpha_values_to_plot:
                        # Construct the path to the results_parameters.csv file
                        parameters_file_path = os.path.join(subfolder_path, 'results_parameters.csv')

                        # Check if the file exists
                        if os.path.exists(parameters_file_path):
                            # Read CSV file into a pandas DataFrame
                            df_parameters = pd.read_csv(parameters_file_path)

                            # Check if any rows are present in the DataFrame
                            if not df_parameters.empty:
                                # Extract peak amplitude and peak frequency values
                                original_peak_amplitude = df_parameters['inital_amp'].values[0]
                                original_peak_frequency = df_parameters['iniial_peak'].values[0]
                                peak_amplitude = df_parameters['Amp_opt'].values[0]
                                peak_frequency = df_parameters['fp_opt_signal'].values[0]

                                # Generate the original power spectrum
                                original_spectrum = PowerSpectrum(alpha_value, 100, constant_temperature, constant_vw)
                                original_signal = original_spectrum.Omega_GW(frequencies, original_spectrum.Amp, original_spectrum.fp_0())
                                original_signals.append(original_signal)

                                # Generate the reconstructed power spectrum (replace with your reconstruction method)
                                reconstructed_spectrum = PowerSpectrum(alpha_value, 100, constant_temperature, constant_vw)
                                reconstructed_signal = reconstructed_spectrum.Omega_GW(frequencies, peak_amplitude, peak_frequency)
                                reconstructed_signals.append(reconstructed_signal)
                                print(reconstructed_signal)

# Plot the original and reconstructed power spectra for each alpha
plt.figure(figsize=(10, 6))
for alpha_value, original_signal, reconstructed_signal in zip(alpha_values_to_plot, original_signals, reconstructed_signals):
    plt.loglog(frequencies, original_signal, label='Original ' r'$\alpha =$' f'{alpha_value}', color='b')
    plt.loglog(frequencies, reconstructed_signal, label='Reconstructed ' r'$\alpha =$' f'{alpha_value}', linestyle = '--', color='r')

# Customize the plot
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$\Omega$')
plt.title('Power Spectrum')
plt.legend()
plt.show()

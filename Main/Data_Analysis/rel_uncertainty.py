import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ps_rewritten import PowerSpectrum  # Replace with the actual import statement for your Powerspectrum class

# Main folder containing subfolders (e.g., DWD_0)
main_folder = 'DWD_0'
# Subfolder for 95_data_points
data_points_subfolder = '95_data_points'
# Temperature to filter
filter_temperature = 180

# Initialize lists to store data
alpha_values = []
vw_values = []
relative_uncertainty_amplitude = []
relative_uncertainty_f_peak = []

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
                if os.path.isdir(subfolder_path) and subfolder_95.startswith('alpha_') and subfolder_95.endswith(f'_temp_{filter_temperature}'):
                    alpha_value = float(subfolder_95.split('_')[1])
                    vw_value = float(subfolder_95.split('_')[3])

                    # Construct the path to the results_parameters.csv file
                    parameters_file_path = os.path.join(subfolder_path, 'results_parameters.csv')

                    # Check if the file exists
                    if os.path.exists(parameters_file_path):
                        # Read CSV file into a pandas DataFrame
                        df_parameters = pd.read_csv(parameters_file_path)

                        # Check if any rows are present in the DataFrame
                        if not df_parameters.empty:
                            # Predict values using PowerSpectrum class
                            powerspectrum = PowerSpectrum(alpha_value, 100, filter_temperature, vw_value)
                            predicted_amplitude = powerspectrum.Amp
                            predicted_f_peak = powerspectrum.fp_0()

                            # Extract original parameters
                            original_amplitude = df_parameters['inital_amp'].values[0]
                            #original_f_peak = df_parameters['iniial_peak'].values[0]

                            # Calculate relative uncertainty for amplitude and f_peak
                            relative_uncertainty_amp = np.abs((predicted_amplitude - original_amplitude) / original_amplitude)
                            #relative_uncertainty_f_peak = np.abs((predicted_f_peak - original_f_peak) / original_f_peak)

                            # Append values to lists
                            alpha_values.append(alpha_value)
                            vw_values.append(vw_value)
                            relative_uncertainty_amplitude.append(relative_uncertainty_amp)
                            #relative_uncertainty_f_peak.append(relative_uncertainty_f_peak)

# Convert lists to arrays
alpha_values = np.array(alpha_values)
vw_values = np.array(vw_values)
relative_uncertainty_amplitude = np.array(relative_uncertainty_amplitude)
relative_uncertainty_f_peak = np.array(relative_uncertainty_f_peak)

# Sort unique alpha values numerically
unique_alpha_values = np.unique(alpha_values)
sorted_unique_alpha_values = np.sort(unique_alpha_values)

# Initialize lists for sorted data
sorted_alpha_values = []
sorted_vw_values = []
sorted_relative_uncertainty_amp_values = []
sorted_relative_uncertainty_f_peak_values = []

# Loop through sorted unique alpha values
for alpha_value in sorted_unique_alpha_values:
    # Find indices where alpha equals the current unique alpha value
    alpha_indices = np.where(alpha_values == alpha_value)[0]

    # Append sorted data for the current alpha value
    sorted_alpha_values.extend(alpha_values[alpha_indices])
    sorted_vw_values.extend(vw_values[alpha_indices])
    sorted_relative_uncertainty_amp_values.extend(relative_uncertainty_amplitude[alpha_indices])
    #sorted_relative_uncertainty_f_peak_values.extend(relative_uncertainty_f_peak[alpha_indices])

# Convert lists to arrays for the sorted data
sorted_alpha_values = np.array(sorted_alpha_values)
sorted_vw_values = np.array(sorted_vw_values)
sorted_relative_uncertainty_amp_values = np.array(sorted_relative_uncertainty_amp_values)
#sorted_relative_uncertainty_f_peak_values = np.array(sorted_relative_uncertainty_f_peak_values)

# Filter data for non-NaN triangles
filtered_indices_amp = ~np.isnan(sorted_relative_uncertainty_amp_values)
filtered_alpha_values_amp = sorted_alpha_values[filtered_indices_amp]
filtered_vw_values_amp = sorted_vw_values[filtered_indices_amp]
filtered_relative_uncertainty_amp_values = sorted_relative_uncertainty_amp_values[filtered_indices_amp]

filtered_indices_f_peak = ~np.isnan(sorted_relative_uncertainty_f_peak_values)
filtered_alpha_values_f_peak = sorted_alpha_values[filtered_indices_f_peak]
filtered_vw_values_f_peak = sorted_vw_values[filtered_indices_f_peak]
#filtered_relative_uncertainty_f_peak_values = sorted_relative_uncertainty_f_peak_values[filtered_indices_f_peak]

# Create contour plots for relative uncertainty of amplitude and f_peak
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
contour_amp = plt.tricontourf(filtered_alpha_values_amp, filtered_vw_values_amp, filtered_relative_uncertainty_amp_values, cmap='plasma')
colorbar_amp = plt.colorbar(contour_amp, label='Relative Uncertainty in Amplitude')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$v_w$')
plt.title(f'Contour Plot of Relative Uncertainty in Amplitude at Temperature {filter_temperature}')

# plt.subplot(1, 2, 2)
# contour_f_peak = plt.tricontourf(filtered_alpha_values_f_peak, filtered_vw_values_f_peak, filtered_relative_uncertainty_f_peak_values, cmap='plasma')
# colorbar_f_peak = plt.colorbar(contour_f_peak, label='Relative Uncertainty in f_peak')
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$v_w$')
# plt.title(f'Contour Plot of Relative Uncertainty in f_peak at Temperature {filter_temperature}')

plt.tight_layout()
plt.show()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Main folder containing subfolders (e.g., DWD_2)
main_folder = 'DWD_2'
# Subfolder for 95_data_points
data_points_subfolder = '95_data_points'
# Temperature to filter
filter_temperature = 180

# Initialize lists to store data
alpha_values = []
vw_values = []
snr_noise_values = []
snr_signal_values = []

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

                    # Add filter for alpha values
                    if 0.05 <= alpha_value <= 0.45:
                        # Construct the path to the aic_results.csv file
                        aic_file_path = os.path.join(subfolder_path, 'results_snr.csv')

                        # Check if the file exists
                        if os.path.exists(aic_file_path):
                            # Read CSV file into a pandas DataFrame
                            df_aic = pd.read_csv(aic_file_path)

                            # Check if any rows are present in the DataFrame
                            if not df_aic.empty:
                                # Extract AIC values for noise and signal
                                snr_noise_value = df_aic['snr_original'].values[0]
                                snr_signal_value = df_aic['snr_reconstructed'].values[0]

                                # Append values to lists
                                alpha_values.append(alpha_value)
                                vw_values.append(vw_value)
                                snr_noise_values.append(snr_noise_value)
                                snr_signal_values.append(snr_signal_value)

# Convert lists to arrays
alpha_values = np.array(alpha_values)
vw_values = np.array(vw_values)
snr_noise_values = np.array(snr_noise_values)
snr_signal_values = np.array(snr_signal_values)

# Sort unique alpha values numerically
unique_alpha_values = np.unique(alpha_values)
sorted_unique_alpha_values = np.sort(unique_alpha_values)

# Initialize lists for sorted data
sorted_alpha_values = []
sorted_vw_values = []
sorted_snr_values = []
sorted2_snr_values = []

# Loop through sorted unique alpha values
for alpha_value in sorted_unique_alpha_values:
    # Find indices where alpha equals the current unique alpha value
    alpha_indices = np.where(alpha_values == alpha_value)[0]

    # Calculate the ratio between AIC noise and AIC signal for the current alpha value
    #aic_ratio_values = aic_noise_values[alpha_indices] / aic_signal_values[alpha_indices]

    # Append sorted data for the current alpha value
    sorted_alpha_values.extend(alpha_values[alpha_indices])
    sorted_vw_values.extend(vw_values[alpha_indices])
    sorted_snr_values.extend(snr_noise_values[alpha_indices])
    sorted2_snr_values.extend(snr_signal_values[alpha_indices])

# Convert lists to arrays for the sorted data
sorted_alpha_values = np.array(sorted_alpha_values)
print(sorted_alpha_values)
sorted_vw_values = np.array(sorted_vw_values)
print(sorted_vw_values)
sorted_snr_values=np.array(sorted_snr_values)
sorted2_snr_values = np.array(sorted2_snr_values)
# Filter data for non-NaN triangles
filtered_indices = ~np.isnan(sorted_snr_values)
filtered_alpha_values = sorted_alpha_values[filtered_indices]
filtered_vw_values = sorted_vw_values[filtered_indices]
filtered_snr_values = sorted_snr_values[filtered_indices]

# Create a contour plot for the ratio between AIC noise and AIC signal
plt.figure(figsize=(8, 6))
contour = plt.tricontourf(filtered_alpha_values, filtered_vw_values, filtered_snr_values, cmap='plasma', vmin=0, vmax=2000)
colorbar = plt.colorbar(contour, label='SNR')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$v_w$')
plt.title(f'Contour Plot of SNR at Temperature {filter_temperature}')
plt.show()

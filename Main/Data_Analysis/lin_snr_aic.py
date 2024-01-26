import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Main folder containing subfolders (e.g., DWD_0)
main_folder = 'DWD_2'
# Subfolder for 95_data_points
data_points_subfolder = '95_data_points'
# Temperature to filter
filter_temperature = 180

# Initialize lists to store data
alpha_values = []
vw_values = []
snr_values = []
aic_signal_values = []
aic_noise_values = []

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

                    # Check if the alpha value is 0.4
                        # Construct the path to the snr_results.csv file
                    snr_file_path = os.path.join(subfolder_path, 'results_snr.csv')
                        # Construct the path to the aic_results.csv file
                    aic_file_path = os.path.join(subfolder_path, 'results_aic.csv')

                        # Check if the files exist
                    if os.path.exists(snr_file_path) and os.path.exists(aic_file_path):
                            # Read CSV files into pandas DataFrames
                            df_snr = pd.read_csv(snr_file_path)
                            df_aic = pd.read_csv(aic_file_path)

                            if not df_snr.empty and not df_aic.empty:
                                # Extract SNR and AIC signal values
                                snr_value = df_snr['snr_original'].values[0]
                                aic_signal_value = df_aic['AIC_signal'].values[0]
                                aic_noise_value = df_aic['AIC_noise'].values[0]

                                # Check if any of the values are NaN
                                if not (np.isnan(snr_value) or np.isnan(aic_signal_value) or np.isnan(aic_noise_value)):
                                    # Append values to lists
                                    alpha_values.append(alpha_value)
                                    vw_values.append(vw_value)
                                    snr_values.append(snr_value)
                                    aic_signal_values.append(aic_signal_value)
                                    aic_noise_values.append(aic_noise_value)

# Convert lists to arrays
alpha_values = np.array(alpha_values)
print(alpha_values)
vw_values = np.array(vw_values)
snr_values = np.array(snr_values)
aic_signal_values = np.array(aic_signal_values)
print(aic_signal_values)
aic_noise_values = np.array(aic_noise_values)
print(aic_noise_values)

# Calculate AIC difference (delta)
 # Replace with the actual AIC noise values
delta = np.array(  aic_noise_values / aic_signal_values)
print(delta)

# Calculate Pearson correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(delta, snr_values)

# Plotting scatterplot
plt.scatter(delta, snr_values, alpha=0.5)
plt.xlabel('AIC Ratio')
plt.ylabel('SNR Values')
plt.grid(True)
plt.title('Scatterplot of AIC Ratio')

# Fit a line to the data using numpy.polyfit
fit_coeffs = np.polyfit(delta, snr_values, 1)
fit_line = np.polyval(fit_coeffs, delta)

# Plot the fit line
plt.plot(delta, fit_line, color='b', linestyle='--', linewidth=2, label='Fit Line')

plt.legend()
plt.show()

# Print the correlation coefficient and p-value
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}")

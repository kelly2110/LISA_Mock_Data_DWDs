import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the specific values for temperature
temperature = 180

# Main folder containing subfolders (e.g., DWD_2)
main_folder = 'DWD_0'
# Subfolder for 95_data_points
data_points_subfolder = '95_data_points'

# Initialize lists to store data
vw_values = []
snr_values_alpha_01 = []
snr_values_alpha_02 = []
snr_values_alpha_03 = []
snr_values_alpha_04 = []

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
                if os.path.isdir(subfolder_path) and subfolder_95.endswith(f'_temp_{temperature}'):
                    alpha_value = float(subfolder_95.split('_')[1])
                    vw_value = float(subfolder_95.split('_')[3])

                    # Construct the path to the snr_results.csv file
                    snr_file_path = os.path.join(subfolder_path, 'results_snr.csv')

                    # Check if the file exists
                    if os.path.exists(snr_file_path):
                        # Read CSV file into a pandas DataFrame
                        df_snr = pd.read_csv(snr_file_path)

                        # Check if any rows are present in the DataFrame
                        if not df_snr.empty:
                            # Append values to lists based on alpha value
                            if vw_value not in vw_values and 0.3 <= vw_value <= 0.9:
                                vw_values.append(vw_value)

                            if alpha_value == 0.1:
                                snr_values_alpha_01.extend(df_snr['snr_original'].values)
                            elif alpha_value == 0.2:
                                snr_values_alpha_02.extend(df_snr['snr_original'].values)
                            elif alpha_value == 0.3:
                                snr_values_alpha_03.extend(df_snr['snr_original'].values)
                            elif alpha_value == 0.4:
                                snr_values_alpha_04.extend(df_snr['snr_original'].values)

# Sort vw_values
vw_values.sort()

# Convert lists to arrays
vw_values = np.array(vw_values)
snr_values_alpha_01 = np.array(snr_values_alpha_01)
snr_values_alpha_02 = np.array(snr_values_alpha_02)
snr_values_alpha_03 = np.array(snr_values_alpha_03)
snr_values_alpha_04 = np.array(snr_values_alpha_04)

# Create a plot with separate lines for different alpha values
plt.figure(figsize=(10, 6))
plt.plot(vw_values, snr_values_alpha_01, linewidth = 0.8, label=r'$\alpha = 0.1$')
plt.plot(vw_values, snr_values_alpha_02, linewidth = 0.8, label=r'$\alpha = 0.2$')
plt.plot(vw_values, snr_values_alpha_03, linewidth = 0.8, label=r'$\alpha = 0.3$')
plt.plot(vw_values, snr_values_alpha_04, linewidth = 0.8, label=r'$\alpha = 0.4$')

plt.xlabel(r'$v_w$')
plt.ylabel('SNR')
plt.title(f'SNR values for Different Alpha Values at Temperature = {temperature}')
plt.legend()
plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Main folder containing subfolders (e.g., DWD_0)
main_folder = 'DWD_2'

# Subfolder for 95_data_points
data_points_subfolder = '95_data_points'

filter_temperature = 180

# Initialize lists to store data
alpha_values = []
vw_values = []
aic_noise_values = []
aic_signal_values = []

# Loop over all subfolders in the main folder
for root, dirs, files in os.walk(main_folder):
    for subfolder in dirs:
        # Check if the subfolder name represents the format {n_c}_data_points
        if subfolder == data_points_subfolder:
            data_points_folder_path = os.path.join(root, subfolder)

            # Loop over subfolders within 95_data_points
            for subfolder_95 in os.listdir(data_points_folder_path):
                subfolder_path = os.path.join(data_points_folder_path, subfolder_95)

                # Check if it's a directory, follows the naming convention, temperature matches, and alpha is below 0.45
                if (
                    os.path.isdir(subfolder_path) and
                    subfolder_95.startswith('alpha_') and
                    subfolder_95.endswith(f'_temp_{filter_temperature}') and
                    float(subfolder_95.split('_')[1]) <= 0.45
                ):
                    alpha_value = float(subfolder_95.split('_')[1])

                    # Construct the path to the aic_results.csv file
                    aic_file_path = os.path.join(subfolder_path, 'results_aic.csv')

                    # Check if the file exists
                    if os.path.exists(aic_file_path):
                        # Read CSV file into a pandas DataFrame
                        df = pd.read_csv(aic_file_path)

                        # Check if any rows are present in the DataFrame
                        if not df.empty:
                            # Extract alpha, vw, and AIC values for noise and signal
                            alpha_values.append(alpha_value)
                            vw_values.append(float(subfolder_95.split('_')[3]))
                            aic_noise_values.append(df['AIC_noise'].values[0])
                            aic_signal_values.append(df['AIC_signal'].values[0])

# Convert lists to arrays
alpha_values = np.array(alpha_values)
vw_values = np.array(vw_values)
aic_noise_values = np.array(aic_noise_values)
aic_signal_values = np.array(aic_signal_values)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for noise-only AIC values
ax.scatter(alpha_values, vw_values, aic_noise_values, c='blue', marker='o', label='AIC Noise')

# Scatter plot for noise + signal AIC values
ax.scatter(alpha_values, vw_values, aic_signal_values, c='red', marker='^', label='AIC Signal')

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$v_w$')
ax.set_zlabel('AIC')
ax.set_title('3D Scatter Plot of AIC Values')
ax.legend()

plt.show()
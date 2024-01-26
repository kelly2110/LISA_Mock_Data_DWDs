import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Main folder containing subfolders (e.g., DWD_1)
main_folder = 'DWD_0'
# Subfolder for 95_data_points
data_points_subfolder = '95_data_points'
# Constant values
constant_temperature = 180

# Initialize lists to store data
alpha_values = []
vw_values = []
mse_values = []

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

                    # Construct the path to the results_parameters.csv file
                    parameters_file_path = os.path.join(subfolder_path, 'results_parameters.csv')

                    # Check if the file exists
                    if os.path.exists(parameters_file_path):
                        # Read CSV file into a pandas DataFrame
                        df_parameters = pd.read_csv(parameters_file_path)

                        # Check if any rows are present in the DataFrame
                        if not df_parameters.empty:
                            # Extract original and reconstructed noise parameters
                            original_noise_params = [3, 15]
                            reconstructed_noise_params = df_parameters[['A_opt_n+s', 'P_opt_n+s']].values[0]

                            # Calculate Mean Squared Error
                            mse = mean_squared_error(original_noise_params, reconstructed_noise_params)

                            # Append values to lists
                            alpha_values.append(alpha_value)
                            vw_values.append(vw_value)
                            mse_values.append(mse)

# Convert lists to arrays
alpha_values = np.array(alpha_values)
vw_values = np.array(vw_values)
mse_values = np.array(mse_values)

# Create a contour plot to visualize the results
plt.figure(figsize=(8, 6))
contour = plt.tricontourf(alpha_values, vw_values, mse_values, cmap='viridis')
plt.colorbar(label='Mean Squared Error (MSE)')
plt.xlabel('Alpha')
plt.ylabel('vw')
plt.title('Mean Squared Error for Noise Parameter Reconstruction (Contour Plot)')
plt.show()

import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
import warnings
import torch
import gc
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for make_example_csv')
parser.add_argument('--config', type=str, help='Path to the configuration file')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)['make_example_csv']

# Set hyperparameters
data_dir = config['data_dir']
columns = config['columns']
num_col_names = config['num_col_names']
mis_num_col_names = config['mis_num_col_names']
dis_col_names = config['dis_col_names']
mis_dis_col_names = config['mis_dis_col_names']
pat_inf = config['pat_inf']
time_seq = config['time_seq']
res_dir = config['res_dir']

# Extract column names
columns = np.load(columns, allow_pickle = True)
num_ind = [index for index, element in enumerate(columns) if element in num_col_names]
num_col_names = columns[num_ind]
mis_num_ind = [index for index, element in enumerate(columns) if element in mis_num_col_names]
mis_num_col_names = columns[mis_num_ind]
dis_ind = [index for index, element in enumerate(columns) if element in dis_col_names]
dis_col_names = columns[dis_ind]
mis_dis_ind = [index for index, element in enumerate(columns) if element in mis_dis_col_names]
mis_dis_col_names = columns[mis_dis_ind]

patient_info = np.load(pat_inf, allow_pickle = True)
time = np.load(time_seq, allow_pickle = True)

# Make duplicate rows in patient info
new_pat_info = np.repeat(patient_info, time, axis=0)

# Get unique patient IDs
unique_IDs = np.unique(new_pat_info[:, 0])

# Calculate the number of patients to select (10% of unique patients)
num_to_select = int(0.05 * len(unique_IDs))

# Randomly select 10% of unique patients
selected_IDs = np.random.choice(unique_IDs, num_to_select, replace=False)

# Find the indices of rows belonging to the selected patients
selected_indices = np.where(np.isin(new_pat_info[:, 0], selected_IDs))

# Extract the selected rows from the original data using the indices
selected_pat_info = new_pat_info[selected_indices]

# Create a dictionary to store the mapping of original IDs to new random IDs
id_mapping = {}

# Generate new random IDs for each original ID
new_IDs = np.arange(len(selected_IDs))  # Generate unique integers as new IDs
np.random.shuffle(new_IDs)             # Shuffle the new IDs randomly

# Populate the dictionary with the mapping
for original_id, new_id in zip(selected_IDs, new_IDs):
    id_mapping[original_id] = new_id

# Create a copy of the data with the new random IDs
selected_pat_info_new = selected_pat_info.copy()

# Replace original IDs with new random IDs in the data
for original_id, new_id in id_mapping.items():
    selected_pat_info_new[selected_pat_info_new[:, 0] == original_id, 0] = new_id
    
# Load numeric data
numeric_data = np.load(data_dir + 'reconstructed_numeric_data.npy', allow_pickle = True)
time_column = numeric_data[selected_indices,0]
numeric_data = numeric_data[:,-91:]
numeric_data_missing = np.load(data_dir + 'reconstructed_missing.npy', allow_pickle = True)
numeric_data = np.where(numeric_data_missing == 1, np.nan, numeric_data)
numeric_data = numeric_data[selected_indices]

# Load discrete data
discrete_data = np.load(data_dir + 'reconstructed_discrete_data_non_missing.npy', allow_pickle = True)
discrete_data = discrete_data[selected_indices]
discrete_data_missing = np.load(data_dir + 'reconstructed_discrete_data_missing.npy', allow_pickle = True)
discrete_data_missing = discrete_data_missing[selected_indices]

# Concatenate
combined_data = np.concatenate((time_column.T,selected_pat_info_new, numeric_data, discrete_data, discrete_data_missing), axis=1)

# Create column names
column_names = ['Time','PID','Center'] + list(num_col_names) + list(dis_col_names) + list(mis_dis_col_names) 

# Convert the NumPy array to a Pandas DataFrame with column names
df = pd.DataFrame(combined_data, columns=column_names)

# Save data
df.to_csv(res_dir + 'reconstructed_sub_data.csv', index=False)
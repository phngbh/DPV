import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
import warnings
import torch
import gc

# Extract column names
columns = np.load('/home/phong.nguyen/data_columns.npy', allow_pickle = True)
search_string = "_missing"
m_indices = [index for index, element in enumerate(columns[92:701]) if search_string in element]
nm_indices = list(set(list(range(609))) - set(m_indices))
columns_discrete_missing = [element for index, element in enumerate(columns[92:701]) if search_string in element]
columns_discrete = [element for index, element in enumerate(columns[92:701]) if index in nm_indices]
columns_numeric = columns[1:92]
columns_numeric_missing = columns[-91:]

patient_info = np.load('/home/phong.nguyen/patient_info.npy', allow_pickle = True)
time = np.load('/home/phong.nguyen/time_sequence.npy', allow_pickle = True)

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
numeric_data = np.load('/home/phong.nguyen/reconstructed_numeric_data.npy', allow_pickle = True)
time_column = numeric_data[selected_indices,0]
numeric_data = numeric_data[:,-91:]
numeric_data_missing = np.load('/home/phong.nguyen/reconstructed_missing.npy', allow_pickle = True)
numeric_data = np.where(numeric_data_missing == 1, np.nan, numeric_data)
numeric_data = numeric_data[selected_indices]

# Load discrete data
discrete_data = np.load('/home/phong.nguyen/reconstructed_discrete_data_non_missing.npy', allow_pickle = True)
discrete_data = discrete_data[selected_indices]
discrete_data_missing = np.load('/home/phong.nguyen/reconstructed_discrete_data_missing.npy', allow_pickle = True)
discrete_data_missing = discrete_data_missing[selected_indices]

# Concatenate
combined_data = np.concatenate((time_column.T,selected_pat_info_new, numeric_data, discrete_data, discrete_data_missing), axis=1)

# Create column names
column_names = ['Time','PID','Center'] + list(columns_numeric) + list(columns_discrete) + list(columns_discrete_missing) 

# Convert the NumPy array to a Pandas DataFrame with column names
df = pd.DataFrame(combined_data, columns=column_names)

# Save data
df.to_csv('reconstructed_sub_data.csv', index=False)
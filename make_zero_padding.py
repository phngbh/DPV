import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
import warnings
import sys
import gc

import preprocessing_functions as uf

print('Start preprocessing data')
print('Starting time: ' + str(datetime.now().time()))

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for preprocessing')
parser.add_argument('--data', type=str, help='File directory to the raw csv data')
parser.add_argument('--var', type=str, help='File directory to variable information (consisting of at least three columns: "Var", "Group" and "Type")')
parser.add_argument('--res_dir', type=str, help='File directory to store the results')
parser.add_argument('--suffix', type=str, help='Name suffix of the results')

args = parser.parse_args()

# Set hyperparameters
data = args.data
var = args.var
res_dir = args.res_dir
suffix = args.suffix

print('Load data')
columns = np.load('/home/phong.nguyen/data_columns.npy', allow_pickle = True)
data = np.load('/home/phong.nguyen/processed_data.npy', allow_pickle = True)

print('Make padding')
padded_data, ori_time = data_padding(data, 50)

print('Make different data segments')
numeric_data = padded_data[:,:,:92]
np.save('/home/phong.nguyen/proc_numeric_data.npy', numeric_data)
missing_mask = padded_data[:,:,-91:]
np.save('/home/phong.nguyen/missing_mask.npy', missing_mask)

discrete_data = padded_data[:,:,92:701]
search_string = "_missing"
m_indices = [index for index, element in enumerate(columns[92:701]) if search_string in element]
nm_indices = list(set(list(range(609))) - set(m_indices))
discrete_data_missing = discrete_data[:,:,m_indices]
discrete_data_non_missing = discrete_data[:,:,nm_indices]
np.save('/home/phong.nguyen/proc_discrete_data_missing.npy', discrete_data_missing)
np.save('/home/phong.nguyen/proc_discrete_data_non_missing.npy', discrete_data_non_missing)






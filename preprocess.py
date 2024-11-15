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
# Load data
clin_typ2 = pd.read_csv(data, sep = ",", header = 0, low_memory=False)
var = pd.read_csv(var, sep = "\t", header = 0, low_memory=False)

print('Recode features')
# Recode and preprocess data
clin_recoded, new_var = uf.feature_recode(clin_typ2, var)
del clin_typ2 
gc.collect()

# Compute mean or mode of the features
mean_mode = uf.mean_mode_calc(clin_recoded, new_var)

print('Preprocess data')
# Preprocessing data
uf.preprocessing_data(data = clin_recoded, feature_info = new_var, mean_mode = mean_mode, result_dir = res_dir, result_suffix = suffix)

print('Finished preprocessing data')
print('Finishing time: ' + str(datetime.now().time()) )

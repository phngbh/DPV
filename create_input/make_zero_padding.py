import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
import warnings
import sys
import gc
import argparse

import preprocessing_functions as pf

def main():
    print('Start preprocessing data')
    print('Starting time: ' + str(datetime.now().time()))

    # Parse arguments
    parser = argparse.ArgumentParser(description='Arguments for preprocessing')
    parser.add_argument('--data', type=str, help='File directory to the preprocessed data')
    parser.add_argument('--columns', type=str, help='File directory to the array of column names")')
    parser.add_argument('--num_col', type=str, help='File directory to the array of numeric column names")')
    parser.add_argument('--mis_num_col', type=str, help='File directory to the array of the names of missing numeric columns")')
    parser.add_argument('--dis_col', type=str, help='File directory to the array of discrete column names")')
    parser.add_argument('--mis_dis_col', type=str, help='File directory to the array of the names of missing discrete columns")')
    parser.add_argument('--seq_len', type=int, help='Sequence length to set for the final data input")')
    parser.add_argument('--res_dir', type=str, help='File directory to store the results')
    parser.add_argument('--suffix', type=str, help='Name suffix of the results')

    args = parser.parse_args()

    # Set hyperparameters
    seq_len = args.seq_len
    res_dir = args.res_dir
    suffix = args.suffix

    print('Load data')
    columns = np.load(args.columns, allow_pickle = True)
    num_col = np.load(args.num_col, allow_pickle = True)
    mis_num_col = np.load(args.mis_num_col, allow_pickle = True)
    dis_col = np.load(args.dis_col, allow_pickle = True)
    mis_dis_col = np.load(args.mis_dis_col, allow_pickle = True)
    data = np.load(args.data, allow_pickle = True)

    print('Make padding')
    padded_data, ori_time = pf.data_padding(data, seq_len)

    print('Make different data segments')
    num_ind = [index for index, element in enumerate(columns) if element in num_col]
    numeric_data = padded_data[:,:,num_ind]
    np.save(res_dir + 'padded_numeric_data.npy', numeric_data)
    mis_num_ind = [index for index, element in enumerate(columns) if element in mis_num_col]
    missing_numeric_data = padded_data[:,:,mis_num_ind]
    np.save(res_dir + 'padded_missing_numeric_data.npy', missing_numeric_data)
    dis_ind = [index for index, element in enumerate(columns) if element in dis_col]
    discrete_data = padded_data[:,:,dis_ind]
    np.save(res_dir + 'padded_discrete_data.npy', discrete_data)
    mis_dis_ind = [index for index, element in enumerate(columns) if element in mis_dis_col]
    missing_discrete_data = padded_data[:,:,mis_dis_ind]
    np.save(res_dir + 'padded_missing_discrete_data.npy', missing_discrete_data)

    print('Finish padding data')
    print('Finishing time: ' + str(datetime.now().time()))

if __name__ == "__main__":
    main()






import numpy as np
import pandas as pd
import torch
import gc
import yaml
import argparse
from datetime import timedelta, datetime
import copy
import os

def main():
    print('Start making input')
    print('Starting time: ' + str(datetime.now().time()))

    # Parse arguments
    parser = argparse.ArgumentParser(description='Arguments for preprocessing')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set hyperparameters
    min_seq_len = config.get('min_seq_len')
    max_seq_len = config.get('max_seq_len')
    columns = config.get('rearranged_columns')
    data = config.get('standardize_data')
    pat_inf = config.get('patient_info')
    target_related_col = config.get('target_related_col')
    target_col = config.get('target_col')
    mis_target_col = config.get('mis_target_col')
    subset = config.get('subset')
    input_dir = config.get('input_dir')
    suffix = config.get('suffix')
    make_sliding_windows = config.get('make_sliding_windows')
    max_prediction_window = config.get('max_prediction_window')
    remove_rows = config.get('remove_rows')

    print("Load data")
    columns = np.load(columns, allow_pickle=True)
    loaded_data = np.load(data, mmap_mode='r')
    proc_data = [loaded_data[key] for key in loaded_data]
    pat_inf = np.load(pat_inf, allow_pickle=True)

    print("Checking for NaNs in proc_data...")
    for idx, arr in enumerate(proc_data):
        if np.isnan(arr).any():
            print(f"...[NaN found] Sample index: {idx}, shape: {arr.shape}")
            nan_locs = np.argwhere(np.isnan(arr))
            print(f"......NaN locations (row, column): {nan_locs}")

    print("Get sequence length")
    time_sequence = [len(arr) for arr in proc_data]
    time_sequence = np.array(time_sequence)

    print("Remove the samples with < minimum time points (seq length)")
    indices_longseq = np.where((time_sequence >= min_seq_len))[0]
    time_sequence = time_sequence[indices_longseq]
    proc_data = [proc_data[i] for i in indices_longseq]
    pat_inf = pat_inf[indices_longseq]
    print(f'The remaining samples are {len(time_sequence)}')

    print("Locate the prediction points")
    last_date_pos = np.cumsum(time_sequence) # Last time points

    print("Indices of non missing target")
    target_missing_info = [array[:, mis_target_col] for array in proc_data] # Extract the column from each array & append to a single list
    target_missing_info = np.concatenate(target_missing_info) # Flatten the list of extracted columns into a single 1D array
    target = [array[:, target_col] for array in proc_data] # Extract the column from each array and append to a single list
    target = np.concatenate(target) # Flatten the list of extracted columns into a single 1D array
    target_missing_info = target_missing_info[last_date_pos - 1] # Get the last time point
    non_missing_indices = np.where((target_missing_info == 0))[0]

    print("...There are " + str(len(non_missing_indices)) + " samples left")

    print("Get last observed target values before the target point")
    proc_data = [proc_data[i] for i in non_missing_indices]
    target_non_missing = [array[:, target_col] for array in proc_data]
    target_missing_info_non_missing = [array[:, mis_target_col] for array in proc_data]
    time_seq = [array[:, 0] for array in proc_data]
    target_last_observed = []
    target_target_point = []
    for i in range(len(target_missing_info_non_missing)):
        missing_info = target_missing_info_non_missing[i]
        indices = np.where((missing_info == 0))[0]
        if len(indices) > 1: 
            pos = indices[-2]
            pos_tar_point = indices[-1]
            tar = target_non_missing[i][pos]
            tar_tar_point = target_non_missing[i][pos_tar_point]
            time = time_seq[i][pos]
            time_las = time_seq[i][pos_tar_point]
            target_last_observed.append([time_las - time, tar])
            target_target_point.append(tar_tar_point)
    np.save(input_dir + "last_observed_target_" + suffix + ".npy",target_last_observed)
    np.save(input_dir + "target_point_target_" + suffix + ".npy",target_target_point)

    print("Filter for non-missing target")
    np.save(input_dir + "/mean_target_" + suffix + ".npy", np.mean(target))
    np.save(input_dir + "/std_target_" + suffix + ".npy", np.std(target))
    target_last = target[last_date_pos-1]
    target_last = target_last[non_missing_indices]

    print("Save patient info")
    pat_inf = pat_inf[non_missing_indices]
    np.save(input_dir + "/patient_info" + suffix + ".npy", pat_inf)

    print("Extract samples with non-missing target")
    time_sequence = time_sequence[non_missing_indices]

    print("Filter out target-related columns")
    target_related_columns = np.load(target_related_col)
    columns_filtered = np.delete(columns, target_related_columns, axis=0)
    indices_filtered = np.where(np.isin(columns, columns_filtered))[0]
    proc_data = [arr[:,indices_filtered] for arr in proc_data]
    print(f"...The final number of features is {len(columns_filtered)}")

    print("Set the seed for reproducibility")
    torch.manual_seed(42)

    print("Randomly choose a subset of samples")
    subset_indices = torch.randperm(len(proc_data))[:subset]

    print("Subset data")
    # padded_data = padded_data[subset_indices,:,:]
    proc_data = [proc_data[i] for i in subset_indices]
    time_sequence = time_sequence[subset_indices]
    target_last = target_last[subset_indices]

    print("Start to process samples")
    remove_rows = list(map(int, remove_rows.split(',')))
    for remove_row in remove_rows:
        
        print("Process samples for prediction point " + str(remove_row))
        new_data = np.zeros([len(proc_data),max_seq_len-1,len(columns_filtered)])
        
        for i in range(len(proc_data)):
            
            print(f'Process sample {i}...')
            data_i = copy.deepcopy(proc_data[i]) # Data of sample i

            # Shift the time column to begin with 0 (to avoid negative values)
            data_i[:,0] -= data_i[:,0].min()
            assert np.all(data_i[:, 0] >= 0), f"Negative time values still exist in sample {i}"

            # Modify the time info
            time_i = copy.deepcopy(data_i[1:,0]) # extract the time column & remove the first 1 dates (not useful for prediction)
            if np.any(time_i < 0) or np.isnan(time_i).any():
                print('There is NaNs or non-positive value in the time column.')
            time_i = np.log10(time_i + 0.1)
            assert not np.isnan(time_i).any(), "NaNs detected in the time column after log transformation." 

            data_i = data_i[:-1,:] # Remove the last 1 rows
            data_i[:,0] = time_i # Change the time column

            # # Standardise the data
            # data_i[:,1:] = (data_i[:,1:] - means.reshape(1,-1))/stds.reshape(1,-1)

            # Make slidding windows
            window_length = len(data_i) - max_prediction_window # Get the window length for sample i
            if bool(make_sliding_windows):
                if len(data_i) >= window_length + remove_row: # Ensure data_i has enough rows
                    # Create the sliding window by removing the specified number of last rows
                    sliding_window = data_i[:-remove_row] if remove_row > 0 else data_i
                    sliding_window = sliding_window[-window_length:]  # Ensure the window length is consistent

                    # Assign the sliding window to new_data
                    new_data[i, -len(sliding_window):, :] = sliding_window
                else:
                    print(f"...data does not have enough rows for the specified window_length and remove_rows.")
            else:
                new_data[i, -len(data_i):, :] = data_i

            del data_i
            gc.collect()
            
        print("...Save samples of prediction point " + str(remove_row))
        torch.save(torch.from_numpy(new_data),input_dir + '/data_subset_' + suffix + '_' + str(remove_row) + ".pth")
        
        del new_data
        gc.collect()

    print("Save rest of data")
    np.save(input_dir + "/columns_" + suffix + ".npy", columns_filtered)
    torch.save(torch.from_numpy(target_last),input_dir + '/target_subset_' + suffix  + '.pth')
    torch.save(torch.from_numpy(time_sequence), input_dir + '/time_sequence_subset_' + suffix  + '.pth')

    print('Finished making input')
    print('Finishing time: ' + str(datetime.now().time()) )

if __name__ == "__main__":
    main()
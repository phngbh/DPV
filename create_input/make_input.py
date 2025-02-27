import numpy as np
import pandas as pd
import torch
import gc
import yaml
import argparse
from datetime import timedelta, datetime

def main():
    print('Start making input')
    print('Starting time: ' + str(datetime.now().time()))

    # Parse arguments
    parser = argparse.ArgumentParser(description='Arguments for preprocessing')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['make_input']

    # Set hyperparameters
    min_seq_len = config['min_seq_len']
    max_seq_len = config['max_seq_len']
    col_names = config['col_names']
    num_col_names = config['num_col_names']
    mis_num_col_names = config['mis_num_col_names']
    dis_col_names = config['dis_col_names']
    mis_dis_col_names = config['mis_dis_col_names']
    data = config['data']
    pat_inf = config['pat_inf']
    target_related_col = config['target_related_col']
    target_col = config['target_col']
    mis_target_col = config['mis_target_col']
    target_name = config['target_name']
    subset = config['subset']
    res_dir = config['res_dir']
    suffix = config['suffix']
    make_sliding_windows = config['make_sliding_windows']
    max_prediction_window = config['max_prediction_window']
    remove_rows = config['remove_rows']

    print("Load data")
    columns = np.load(col_names, allow_pickle=True)
    columns_numeric = np.load(num_col_names, allow_pickle=True)
    columns_numeric_missing = np.load(mis_num_col_names, allow_pickle=True)
    columns_discrete = np.load(dis_col_names, allow_pickle=True)
    columns_discrete_missing = np.load(mis_dis_col_names, allow_pickle=True)
    proc_data = np.load(data, allow_pickle=True)
    pat_inf = np.load(pat_inf, allow_pickle=True)

    print("Get sequence length")
    time_sequence = [len(arr) for arr in proc_data]
    time_sequence = np.array(time_sequence)

    print("Remove the samples with < minimum time points (seq length)")
    indices_longseq = np.where((time_sequence >= min_seq_len))[0]
    time_sequence = time_sequence[indices_longseq]
    proc_data = [proc_data[i] for i in indices_longseq]
    pat_inf = pat_inf[indices_longseq]
    print(f'The remaining samples are {len(time_sequence)}')

    print("Filter out target-related columns")
    target_related_columns = np.load(target_related_col)
    columns = np.delete(columns, target_related_columns, axis=0)

    print("Get the columns indices")
    indices_numeric = np.where(np.isin(columns, columns_numeric))[0]
    indices_numeric_missing = np.where(np.isin(columns, columns_numeric_missing))[0]
    indices_discrete = np.where(np.isin(columns, columns_discrete))[0]
    indices_discrete_missing = np.where(np.isin(columns, columns_discrete_missing))[0]

    print("Get means and standard deviations for numeric data")
    numeric_data = [array[:, indices_numeric] for array in proc_data]
    numeric_data = np.concatenate(numeric_data, axis=0)
    means_numeric = np.mean(numeric_data, axis=0)
    stds_numeric = np.std(numeric_data, axis=0)
    nonzero_stds_indices_numeric = np.where((stds_numeric != 0))[0] # Look for non-uniform variables
    # Filter for non-uniform variables
    indices_numeric = indices_numeric[nonzero_stds_indices_numeric]
    means_numeric = means_numeric[nonzero_stds_indices_numeric][1:] # Without the time column
    stds_numeric = stds_numeric[nonzero_stds_indices_numeric][1:] # Without the time column
    np.save(res_dir + "/means_numeric" + suffix + ".npy", means_numeric)
    np.save(res_dir + "/stds_numeric" + suffix + ".npy", stds_numeric)
    print("...The numeric column length is " + str(len(indices_numeric)))
    del numeric_data
    gc.collect()

    print("Get means and standard deviations for missing numeric data")
    numeric_data_missing = [array[:, indices_numeric_missing] for array in proc_data]
    numeric_data_missing = np.concatenate(numeric_data_missing, axis=0)
    means_numeric_missing = np.mean(numeric_data_missing, axis=0)
    stds_numeric_missing = np.std(numeric_data_missing, axis=0)
    nonzero_stds_indices_numeric_missing = np.where((stds_numeric_missing != 0))[0] # Look for non-uniform variables
    # Filter for non-uniform variables
    indices_numeric_missing = indices_numeric_missing[nonzero_stds_indices_numeric_missing]
    means_numeric_missing = means_numeric_missing[nonzero_stds_indices_numeric_missing] 
    stds_numeric_missing = stds_numeric_missing[nonzero_stds_indices_numeric_missing]
    np.save(res_dir + "/means_numeric_missing" + suffix + ".npy", means_numeric_missing)
    np.save(res_dir + "/stds_numeric_missing" + suffix + ".npy", stds_numeric_missing)
    print("...The missing numeric column length is " + str(len(indices_numeric_missing)))
    del numeric_data_missing
    gc.collect()

    print("Get means and standard deviations for discrete data")
    discrete_data = [array[:, indices_discrete] for array in proc_data]
    discrete_data = np.concatenate(discrete_data, axis=0)
    means_discrete = np.mean(discrete_data, axis=0)
    stds_discrete = np.std(discrete_data, axis=0)
    nonzero_stds_indices_discrete = np.where((stds_discrete != 0))[0] # Look for non-uniform variables
    # Filter for non-uniform variables
    indices_discrete = indices_discrete[nonzero_stds_indices_discrete]
    means_discrete = means_discrete[nonzero_stds_indices_discrete]
    stds_discrete = stds_discrete[nonzero_stds_indices_discrete] 
    np.save(res_dir + "/means_discrete" + suffix + ".npy", means_discrete)
    np.save(res_dir + "/stds_discrete" + suffix + ".npy", stds_discrete)
    print("...The discrete column length is " + str(len(indices_discrete)))
    del discrete_data
    gc.collect()

    print("Get means and standard deviations for missing discrete data")
    discrete_data_missing = [array[:, indices_discrete_missing] for array in proc_data]
    discrete_data_missing = np.concatenate(discrete_data_missing, axis=0)
    means_discrete_missing = np.mean(discrete_data_missing, axis=0)
    stds_discrete_missing = np.std(discrete_data_missing, axis=0)
    nonzero_stds_indices_discrete_missing = np.where((stds_discrete_missing != 0))[0] # Look for non-uniform variables
    # Filter for non-uniform variables
    indices_discrete_missing = indices_discrete_missing[nonzero_stds_indices_discrete_missing]
    means_discrete_missing = means_discrete_missing[nonzero_stds_indices_discrete_missing]
    stds_discrete_missing = stds_discrete_missing[nonzero_stds_indices_discrete_missing] 
    np.save(res_dir + "/means_discrete_missing" + suffix + ".npy", means_discrete_missing)
    np.save(res_dir + "/stds_discrete_missing" + suffix + ".npy", stds_discrete_missing)
    print("...The missing discrete column length is " + str(len(indices_discrete_missing)))
    del discrete_data_missing
    gc.collect()

    print("Rearrange the columns")
    columns = np.concatenate((columns[indices_numeric], columns[indices_discrete], columns[indices_numeric_missing], 
                            columns[indices_discrete_missing]), axis=0)
    print("...The final column length is " + str(len(columns)))

    print("Locate the prediction points")
    last_date_pos = np.cumsum(time_sequence) # Last time points

    print("Indices of non missing target")
    # proc_data = np.concatenate(proc_data, axis = 0) # concatenate the list of arrays to a 2d array
    target_missing_info = [array[:, mis_target_col] for array in proc_data] # Extract the column from each array & append to a single list
    target_missing_info = np.concatenate(target_missing_info) # Flatten the list of extracted columns into a single 1D array
    target = [array[:, target_col] for array in proc_data] # Extract the column from each array and append to a single list
    target = np.concatenate(target) # Flatten the list of extracted columns into a single 1D array
    if target_name == 'hba1c':
        zero_target_indices = np.where((target == 0))[0] # Locate where the target is 0 (i.e. NaNs(only for HbA1C))
        target_missing_info[zero_target_indices] = 1 # Change the missing info at these location to be 1 (ie. NaNs(only for HbA1C))
        target[zero_target_indices] = np.mean(np.delete(target,zero_target_indices)) # Fill the zero pos with mean (only for HbA1C)
    elif target_name == 'ldl':
        outlier_target_indices = np.where((target >= 300))[0] # Locate where the target is very high (i.e. NaNs(only for LDL))
        target_missing_info[outlier_target_indices] = 1 # Change the missing info at these location to be 1 (ie. NaNs(only for LDL))
        target[outlier_target_indices] = np.mean(np.delete(target,outlier_target_indices)) # Fill the outlier pos with mean (only for LDL)
    target_missing_info = target_missing_info[last_date_pos - 1] # Get the last time point
    #target_missing_info = proc_data[last_date_pos - 1, 739]
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
    np.save(res_dir + "last_observed_target" + suffix + ".npy",target_last_observed)
    np.save(res_dir + "target_point_target" + suffix + ".npy",target_target_point)

    print("Filter for non-missing target")
    # target = proc_data[:,39]
    # target = uf.make_data_bins(target)
    np.save(res_dir + "mean" + suffix + ".npy", np.mean(target))
    np.save(res_dir + "std" + suffix + ".npy", np.std(target))
    target = (target - np.mean(target))/np.std(target) # Standardise the target
    target_last = target[last_date_pos-1]
    target_last = target_last[non_missing_indices]

    print("Save patient info")
    pat_inf = pat_inf[non_missing_indices]
    np.save(res_dir + "/patient" + suffix + ".npy", pat_inf)

    # del proc_data
    # gc.collect()

    print("Extract samples with non-missing target")
    # proc_data = [proc_data[i] for i in non_missing_indices]
    time_sequence = time_sequence[non_missing_indices]

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
    for remove_row in range(len(remove_rows)):
        
        print("Process samples for prediction point " + str(remove_row))
        new_data = np.zeros([len(proc_data),max_seq_len-1,len(columns)])
        
        for i in range(len(proc_data)):

            data_i = proc_data[i] # Data of sample i

            # Modify the time info
            time_i = data_i[1:,0] # extract the time column & remove the first 1 dates (not useful for prediction)
            time_i = np.log10(time_i + 0.1) # Log10 transform the time info
            data_i = data_i[:-1,:] # Remove the last 1 rows
            data_i[:,0] = time_i # Change the time column

            # Get sliding windows
            #data_i = data_i[:-4,:]

            # Rearrange the data columns
            numeric_data_i = data_i[:,indices_numeric]
            numeric_data_i[:,1:] = (numeric_data_i[:,1:] - means_numeric)/stds_numeric # Standardise the numeric data
            discrete_data_i = data_i[:,indices_discrete]
            discrete_data_i = (discrete_data_i - means_discrete)/stds_discrete # Standardise the discrete data
            numeric_data_missing_i = data_i[:,indices_numeric_missing]
            numeric_data_missing_i = (numeric_data_missing_i - means_numeric_missing)/stds_numeric_missing # Standardise the numeric data
            discrete_data_missing_i = data_i[:,indices_discrete_missing]
            discrete_data_misisng_i = (discrete_data_missing_i - means_discrete_missing)/stds_discrete_missing # Standardise the numeric data
            data_i = np.concatenate((numeric_data_i, discrete_data_i, numeric_data_missing_i, discrete_data_missing_i), axis=1)

            # Make slidding windows
            window_length = len(data_i) - max_prediction_window # Get the window length for sample i
            if make_sliding_windows:
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

            # new_data.append(data_i)
            # new_data[i,-len(data_i):,:] = data_i
            del data_i, numeric_data_i, numeric_data_missing_i, discrete_data_i, discrete_data_missing_i
            gc.collect()
            
        print("...Save samples of prediction point " + str(remove_row))
        torch.save(torch.from_numpy(new_data),res_dir + '/data_subset_' + suffix + '_' + str(remove_row) + ".pth")
        
        del new_data
        gc.collect()

    print("Save rest of data")
    np.save(res_dir + "/columns_rearranged_" + suffix + ".npy", columns)
    torch.save(torch.from_numpy(target_last),res_dir + '/target_subset_' + suffix + '.pth')
    torch.save(torch.from_numpy(time_sequence), res_dir + '/time_sequence_subset_' + suffix + '.pth')

    print('Finished making input')
    print('Finishing time: ' + str(datetime.now().time()) )

if __name__ == "__main__":
    main()

    
    

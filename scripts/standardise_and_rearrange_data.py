import numpy as np
import yaml
import argparse
import os
import sys

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(cfg):

    target_name = cfg.get('target_name')
    output_dir = cfg.get('output_dir', '.')
    os.makedirs(output_dir, exist_ok=True)

    print('Load data')
    columns = np.load(cfg.get('columns'), allow_pickle=True)
    columns_numeric = np.load(cfg.get('columns_numeric'), allow_pickle=True)
    columns_numeric_missing = np.load(cfg.get('columns_numeric_missing'), allow_pickle=True)
    columns_discrete = np.load(cfg.get('columns_discrete'), allow_pickle=True)
    columns_discrete_missing = np.load(cfg.get('columns_discrete_missing'), allow_pickle=True)
    target_related_columns = np.load(cfg.get('target_related_columns'), allow_pickle=True)
    proc_data = list(np.load(cfg.get('data'), allow_pickle=True))

    print('The sample size is ', len(proc_data))

    print('Rearrange the data columns')
    numeric_indices = np.where(np.isin(columns, columns_numeric))[0]
    numeric_missing_indices = np.where(np.isin(columns, columns_numeric_missing))[0]
    discrete_indices = np.where(np.isin(columns, columns_discrete))[0]
    discrete_missing_indices = np.where(np.isin(columns, columns_discrete_missing))[0]
    rearranged_indices = np.concatenate([numeric_indices, discrete_indices, numeric_missing_indices, discrete_missing_indices])

    print('Get some global measurements')
    n_cols = proc_data[0].shape[1]
    total_sum = np.zeros(n_cols)
    total_sq_sum = np.zeros(n_cols)
    total_count = 0
    for arr in proc_data:
        total_sum += arr.sum(axis=0)
        total_sq_sum += (arr ** 2).sum(axis=0)
        total_count += arr.shape[0]
    global_mean = total_sum / total_count
    global_var = (total_sq_sum / total_count) - (global_mean ** 2)
    global_std = np.sqrt(global_var)
    nonzero_std_ind = np.where((global_std != 0))[0]

    print('Remove the uniform variables')
    rearranged_indices = rearranged_indices[np.isin(rearranged_indices, nonzero_std_ind)]
    columns_rearranged = columns[rearranged_indices]
    proc_data = [arr[:, rearranged_indices] for arr in proc_data]
    print('The final number of features is ', len(rearranged_indices))

    print('Standardise numeric data')
    numeric_indices = numeric_indices[np.isin(numeric_indices, rearranged_indices)]
    numeric_mean = global_mean[numeric_indices]
    numeric_std = global_std[numeric_indices]
    for i in range(len(proc_data)):
        numeric_arr = proc_data[i][:,numeric_indices]
        numeric_arr[:,1:] = (numeric_arr[:,1:] - numeric_mean[1:])/numeric_std[1:]
        proc_data[i][:,numeric_indices] = numeric_arr

    print('Get the target-related columns')
    # target_related_columns may be indices or names; handle both
    if np.issubdtype(target_related_columns.dtype, np.integer):
        # integer indices
        target_related_names = columns[target_related_columns]
    else:
        # assume names
        target_related_names = np.array(target_related_columns, dtype=columns.dtype)

    print('Target related variables are: ', target_related_names)
    new_target_related_columns = np.where(np.isin(columns_rearranged, target_related_names))[0]

    print('Save results')
    data_out_path = os.path.join(output_dir, 'data_list.npz')
    np.savez_compressed(data_out_path, *proc_data)
    np.save(os.path.join(output_dir, 'columns.npy'), columns_rearranged)
    np.save(os.path.join(output_dir, target_name + 'target_related_columns.npy'), new_target_related_columns)
    np.save(os.path.join(output_dir, target_name + 'target_related_columns_names.npy'), target_related_names)
    print('Saved to', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standardise and rearrange dataset using config.yml')
    parser.add_argument('--config', '-c', default='config.yml', help='Path to YAML config file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f'Config file not found: {args.config}', file=sys.stderr)
        sys.exit(2)
    config = load_config(args.config)
    main(config)
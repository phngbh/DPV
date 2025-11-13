import os
import gc
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

def load_npy(path, allow_pickle=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path, allow_pickle=allow_pickle)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def map_ids_and_replace(selected_pat_info, original_ids, new_ids):
    # vectorized mapping for ID replacement
    mapping = dict(zip(original_ids, new_ids))
    ids_col = selected_pat_info[:, 0]
    # create vectorized replacement using numpy where for each unique id
    out = selected_pat_info.copy()
    for orig, new in mapping.items():
        out[ids_col == orig, 0] = new
    return out

def main():
    parser = argparse.ArgumentParser(description='Create example CSV from reconstructed synthetic data')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_all = yaml.safe_load(f)
    cfg = cfg_all.get('make_example_csv', cfg_all)

    # config entries (required)
    columns_path = cfg['columns']
    pat_inf_path = cfg['pat_inf']
    time_seq_path = cfg['time_seq']
    data_dir = cfg['data_dir']          # can be directory or prefix
    res_dir = cfg.get('res_dir', './results/')
    subset_fraction = float(cfg.get('subset_fraction', 0.05))
    random_seed = int(cfg.get('seed', 42))

    # reconstructed file names (allow overrides)
    recon_numeric_file = cfg.get('reconstructed_numeric', os.path.join(data_dir, 'reconstructed_numeric_data.npy'))
    recon_numeric_missing_file = cfg.get('reconstructed_numeric_missing', os.path.join(data_dir, 'reconstructed_missing.npy'))
    recon_discrete_file = cfg.get('reconstructed_discrete_non_missing', os.path.join(data_dir, 'reconstructed_discrete_data_non_missing.npy'))
    recon_discrete_missing_file = cfg.get('reconstructed_discrete_missing', os.path.join(data_dir, 'reconstructed_discrete_data_missing.npy'))

    ensure_dir(res_dir)

    print(f"[{datetime.now()}] Loading metadata and reconstructed arrays")
    columns = load_npy(columns_path)
    patient_info = load_npy(pat_inf_path)
    time = load_npy(time_seq_path)

    # Ensure time is 1-D integer array with length matching patient_info rows
    time = np.asarray(time).astype(int).ravel()
    if patient_info.shape[0] != time.shape[0]:
        raise ValueError("patient_info rows and time_seq length must match")

    # Build repeated patient-info rows for each time step (legacy behavior)
    new_pat_info = np.repeat(patient_info, time, axis=0)

    # Select a subset of unique patient IDs to include
    unique_IDs = np.unique(new_pat_info[:, 0])
    num_to_select = max(1, int(subset_fraction * len(unique_IDs)))
    rng = np.random.default_rng(random_seed)
    selected_IDs = rng.choice(unique_IDs, size=num_to_select, replace=False)

    # Rows corresponding to selected patients
    selected_mask = np.isin(new_pat_info[:, 0], selected_IDs)
    selected_indices = np.where(selected_mask)[0]
    selected_pat_info = new_pat_info[selected_mask]

    # Create new random IDs (0..num_to_select-1) and replace original IDs
    new_IDs = np.arange(num_to_select)
    rng.shuffle(new_IDs)
    # map original selected_IDs -> new_IDs
    selected_pat_info_new = map_ids_and_replace(selected_pat_info, selected_IDs, new_IDs)

    print(f"[{datetime.now()}] Loading reconstructed data arrays")
    numeric_data = load_npy(recon_numeric_file)
    numeric_missing = load_npy(recon_numeric_missing_file)
    discrete_data = load_npy(recon_discrete_file)
    discrete_missing = load_npy(recon_discrete_missing_file)

    # Subselect same rows for reconstructed arrays
    # If reconstructed arrays are 3D (N, T, F) or 2D (N, F), we try to select rows accordingly.
    def select_rows(arr, idx):
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return np.array([])
        if arr.shape[0] >= len(new_pat_info) and len(idx) > 0:
            # arr indexed per repeated rows
            return arr[idx]
        elif arr.shape[0] == patient_info.shape[0]:
            # arr given per original patient; expand to repeated rows then select
            repeated = np.repeat(arr, time, axis=0)
            return repeated[idx]
        else:
            # fallback: attempt direct selection (may raise)
            return arr[idx]

    numeric_selected = select_rows(numeric_data, selected_indices)
    numeric_missing_selected = select_rows(numeric_missing, selected_indices)
    discrete_selected = select_rows(discrete_data, selected_indices)
    discrete_missing_selected = select_rows(discrete_missing, selected_indices)

    # Derive time column if present: prefer first column of numeric_selected if that encodes time,
    # else use repeated time values for selected_indices
    if numeric_selected.size == 0:
        time_column = np.empty((len(selected_indices), 0))
    else:
        if numeric_selected.ndim > 1 and numeric_selected.shape[1] > 0 and cfg.get('time_in_numeric_first_col', False):
            time_column = numeric_selected[:, 0:1]
            # drop that column from numeric features
            numeric_selected = numeric_selected[:, 1:]
            numeric_missing_selected = numeric_missing_selected[:, 1:] if numeric_missing_selected.ndim > 1 else numeric_missing_selected
        else:
            # build a time column from repeated time list
            repeated_time = np.repeat(time, time, axis=0) if False else None  # intentionally not used; instead derive per-row lengths
            # simpler: construct single scalar time length for each selected row from new_pat_info (we have no per-row timestamp here)
            # Use original 'time' per patient expanded to repeated rows:
            patient_ids_per_row = np.repeat(np.arange(len(time)), time)
            # patient_ids_per_row length should equal new_pat_info length
            if patient_ids_per_row.shape[0] == new_pat_info.shape[0]:
                time_vals = time[ new_pat_info[selected_mask, 0].astype(int) ]
                time_column = time_vals.reshape(-1, 1)
            else:
                # fallback to zeros
                time_column = np.zeros((len(selected_indices), 1), dtype=int)

    # Prepare column name lists: try to derive from columns file if names provided there
    all_columns = np.asarray(columns)
    num_col_list = cfg.get('num_col_names', [])
    mis_num_col_list = cfg.get('mis_num_col_names', [])
    dis_col_list = cfg.get('dis_col_names', [])
    mis_dis_col_list = cfg.get('mis_dis_col_names', [])

    # If config provided names are empty, try to infer from 'columns' array by membership
    def pick_names(requested_names):
        if requested_names:
            return list(requested_names)
        return []

    num_names = pick_names(num_col_list)
    dis_names = pick_names(dis_col_list)
    mis_dis_names = pick_names(mis_dis_col_list)

    # Concatenate horizontally: [time_column, selected_pat_info_new, numeric_selected, discrete_selected, discrete_missing_selected]
    parts = []
    if time_column is not None and time_column.size > 0:
        parts.append(time_column)
    parts.append(selected_pat_info_new.astype(object))  # patient info may contain strings or mixed types
    if numeric_selected.size > 0:
        parts.append(np.asarray(numeric_selected))
    if discrete_selected.size > 0:
        parts.append(np.asarray(discrete_selected))
    if discrete_missing_selected.size > 0:
        parts.append(np.asarray(discrete_missing_selected))

    combined = np.concatenate(parts, axis=1)

    # Build column headers: attempt minimal safe naming
    pid_center_cols = []
    if selected_pat_info_new.shape[1] >= 2:
        pid_center_cols = ['PID', 'Center'] if selected_pat_info_new.shape[1] >= 2 else ['PID']
    elif selected_pat_info_new.shape[1] == 1:
        pid_center_cols = ['PID']
    header = []
    if time_column is not None and time_column.size > 0:
        header.append('Time')
    header.extend(pid_center_cols)
    # append numeric / discrete names using provided lists or generic placeholders
    n_num = numeric_selected.shape[1] if numeric_selected.ndim > 1 else 0
    n_dis = discrete_selected.shape[1] if discrete_selected.ndim > 1 else 0
    n_mis_dis = discrete_missing_selected.shape[1] if discrete_missing_selected.ndim > 1 else 0

    if num_names and len(num_names) >= n_num:
        header.extend(num_names[:n_num])
    else:
        header.extend([f'num_{i}' for i in range(n_num)])
    if dis_names and len(dis_names) >= n_dis:
        header.extend(dis_names[:n_dis])
    else:
        header.extend([f'dis_{i}' for i in range(n_dis)])
    if mis_dis_names and len(mis_dis_names) >= n_mis_dis:
        header.extend(mis_dis_names[:n_mis_dis])
    else:
        header.extend([f'dis_missing_{i}' for i in range(n_mis_dis)])

    # Create DataFrame - convert combined to numeric where possible
    try:
        df = pd.DataFrame(combined, columns=header)
    except Exception:
        # fall back: coerce to string columns
        df = pd.DataFrame(combined.astype(object))
        df.columns = header[:df.shape[1]]

    out_file = os.path.join(res_dir, cfg.get('out_filename', 'reconstructed_sub_data.csv'))
    df.to_csv(out_file, index=False)
    print(f"[{datetime.now()}] Saved CSV to {out_file}")

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
import warnings
import torch
import os

def feature_recode(data, feature_info):
    """
    Manually recode the features to the same encoding based on their types and meaning 
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data to cluster.
    feature_info: dataframe, shape (n_features,)
        Feature info, consisting of at least three columns: 'Var', 'Group' and 'Type'
    
    Returns
    -------
    Newly recoded data and feature_info
    """
    
    # Remove columns with all NaNs
    data = data.dropna(axis=1, how='all')
    
    #Filter variables
    feature_info = feature_info.loc[feature_info['Var'].isin(data.columns),:]
    
    # Define the mapping of values to be recoded
    value_mapping = {
        -2: np.NaN,
        -1: np.NaN,
        0: np.NaN,
        'j': 1,
        'n': 0,
        'J': 1,
        'N': 0,
        'F': 0
    }
    
    print('Processing date variables')
    # Process some date columns
    data['auf_dat'] = pd.to_datetime(data['auf_dat'], format = "%m/%d/%Y")
    # data = data.drop('bdau', axis=1)
    # data = data.drop('entlasstag', axis=1)
    # data = data.drop('jahr', axis=1)
    # data = data.drop('DIA_DAT', axis=1)
    
    print('Recoding demographics and life style variables')
    
    # Extract variable and type vectors
    var = feature_info.loc[feature_info['Group'].isin(['Demographics','Life style']), 'Var']
    typ = feature_info.loc[feature_info['Group'].isin(['Demographics','Life style']), 'Type']

    # Extract categorical vars
    ind = typ[typ == 'Categorical'].index
    var_cat = var[ind]

    for v in var_cat:
        if data[v].nunique() > 2:
            # Recode the values in 'column_name' based on the mapping
            data[v] = data[v].replace(value_mapping)

    # Remove correlated variable
    # data = data.drop(['altjahr','kind_dn','mutter_dn','vater_dn','ausland','year_of_birth','LANDKIND','LANDMUTT','LANDVATE','KINDSEIT','MUTTERSE','VATERSEI'], axis=1)
    # Remove other variables
    # data = data.drop(['patplz_a','klinplz_a','educated'], axis=1)
    
    print('Recoding diabetes types and diagnosis variables')
    
    # Extract variable and type vectors
    var = feature_info.loc[feature_info['Group'].isin(['Diabetes types','Diagnosis']), 'Var']
    
    for v in var:
        # Recode the values based on the mapping
        data[v] = data[v].replace(value_mapping)
        
        # Remove NaN variables
        # if len(data[np.isnan(data[v])]) == len(data):
        #     data = data.drop(v, axis = 1)
        # if len(data[data[v] == 1]) + len(data[np.isnan(data[v])]) == len(data):
        #     # Recode NaNs
        #     data[v] = data[v].replace({np.NaN: 0})
        # else:
        #     # Recode the values based on the mapping
        #     data[v] = data[v].replace(value_mapping)
            
    print('Recoding treatment variables')
    
    data['inject'] = np.where(data['inject'] == 0.5, np.NaN, data['inject'])
    
    var = feature_info.loc[feature_info['Group'].isin(['Treatment/Management','Procedure']), 'Var']
    typ = feature_info.loc[feature_info['Group'].isin(['Treatment/Management','Procedure']), 'Type']

    # Extract categorical vars
    ind = typ[typ == 'Categorical'].index
    var_cat = var[ind]
    
    for v in var_cat:
        if v == 'INSUL_DA':
            # Fix some typos 
            data[v] = data[v].astype(str)
            data[v] = data[v].str.replace('3013','2013')
            data[v] = data[v].str.replace('2611','2011')
            data[v] = data[v].str.replace('2303','2003')
            # Define the multiple datetime formats
            date_formats = ["%m/%d/%Y",'%Y-%m-%d']
            # Convert 'datetime_column' to datetime with multiple formats
            for date_format in date_formats:
                try:
                    data[v] = pd.to_datetime(data[v], format=date_format)
                    break  # Exit the loop if conversion is successful
                except ValueError:
                    pass  # Continue to the next format if conversion fails
            # Make duration columns
            data['duration_' + v] = data['auf_dat'] - data[v]
            data['duration_' + v] = data['duration_' + v].dt.days
            data = data.drop(v, axis=1)
            ## Recode NaNs to 0
            #data.loc[np.isnan(data['duration_' + v]),'duration_' + v] = 0
        
        elif v == 'ORAL_DAT':
            # Covert to date 
            data[v] = pd.to_datetime(data[v], format = "%m/%d/%Y")
            # Make duration columns
            data['duration_' + v] = data['auf_dat'] - data[v]
            data['duration_' + v] = data['duration_' + v].dt.days
            data = data.drop(v, axis=1)
            
        # elif len(data[np.isnan(data[v])]) == len(data):
        #     data = data.drop(v, axis = 1)
        # elif len(data[data[v] == 1]) + len(data[np.isnan(data[v])]) == len(data):
        #     # Recode NaNs to 0
        #     data[v] = data[v].replace({np.NaN: 0})
        else:
            # Recode the values based on the mapping
            data[v] = data[v].replace(value_mapping)
            
    print('Recoding physiological variables')
    
    # Remove redundant blood pressure vars
    # data = data.drop(['rrsys_e_sds','rrdia_e_sds','rrsys_sds_kiggs','rrdia_sds_kiggs','rrsys_sds_4th','rrdia_sds_4th','eSBP_kiggs','eDBP_kiggs','eSBP_4th','eDBP_4th'], axis = 1)
    
    # Remove redundant HbA1c vars
    # data = data.drop(['hbamom_hd','hbamom_ifcc','hbamom_p','hbasds_hd'], axis = 1)
    
    # Remove redundant BMI vars
    # data = data.drop(['bmisdsk','bmisdski','bmisdskiggs','bmisdswho','bmisdsnvz','bmisdsiotf'], axis = 1)
    
    # Remove redundant height vars
    # data = data.drop(['grossdskiggs','grossdswho'], axis = 1)
    
    # Remove redundant weight vars
    # data = data.drop(['gewsdsk','gewsdswho'], axis = 1)
    
    # Recode some measurements (0 -> NaN)
    data['TEST_L'] = data['TEST_L'].replace(value_mapping) # Testicular volume
    data['TEST_R'] = data['TEST_R'].replace(value_mapping)
    data['KN_AL'] = data['KN_AL'].replace(value_mapping) # Bone age
    
    # Make new LDL classification
    data['LDL_c'] = np.where(data['ldlee'] == 1, 3, 
                            np.where(data['ldle'] == 1, 2,
                                    np.where(data['ldlm'] == 1, 1, 
                                            np.where(np.isnan(data['ldlm']),np.NaN, 0))))
    
    # Make albumin-to-creatinine ratio (uACR)
    data['KREA'] = data['KREA']/1000 # convert mg/dl to g/dl
    data['URIN_ALB'] = np.where(data['ualb_einheit'] == 'mg/24h', data['URIN_ALB']/1440*1000*3/2,data['URIN_ALB']) # convert mg/24h to µg/min and then to mg/dl
    data['URIN_ALB'] = np.where(data['ualb_einheit'] == 'mg/l', data['URIN_ALB']/10,data['URIN_ALB']) # convert mg/l to mg/dl
    data['URIN_ALB'] = np.where(data['ualb_einheit'] == 'µg/min', data['URIN_ALB']*3/2,data['URIN_ALB']) # convert ug/min to mg/dl
    data['uACR'] = np.where(data['ualb_einheit'].isin(['mg/24h','mg/l','mg/dl','µg/min']), data['URIN_ALB']/data['KREA'], np.NaN) # compute uACR (mg/g)
    data['uACR'] = np.where(data['ualb_einheit'] == 'mg/molKrea', data['URIN_ALB']/113.12, data['uACR']) # convert mg/mol to mg/g
    data['uACR'] = np.where(data['ualb_einheit'] == 'mg/mmolKrea', data['URIN_ALB']/113.12/1000, data['uACR']) # convert mg/mmol to mg/g
    data['uACR'] = np.where(data['ualb_einheit'].isna(), np.NaN, data['uACR']) # missing data
    data['uACR'] = np.where(data['ualb_einheit'] == 'mis', np.NaN, data['uACR']) # missing data
    data['uACR_c'] = np.where(data['uACR'] >= 300, 2,
                              np.where(data['uACR'] >= 30, 1, 
                                       np.where(data['uACR'] < 30, 0, np.NaN))) # classifying albuminurea
    
    # Extract variable and type vectors
    var = feature_info.loc[feature_info['Group'] == 'Physiological status', 'Var']
    typ = feature_info.loc[feature_info['Group'] == 'Physiological status', 'Type']
    # Extract categorical vars
    ind = typ[typ == 'Categorical'].index
    var_cat = var[ind]
    
    for v in var_cat:
        # Remove redundant LDL vars
        if v in ['ldlm','ldle','ldlee']:
            data = data.drop(v, axis = 1)
        # Remove redundant ualb var
        if v == 'ualb_einheit':
            data = data.drop(v, axis = 1)
        # Remove redundant blood pressure vars
        if v in ['rre_2nd','rre_2nd_c','rre_4th','rre_4th_c', 'rre_4thn','rre_4thn_c','rre_e','rre_e_c','rre']:
            data = data.drop(v, axis = 1)
        # elif len(data[np.isnan(data[v])]) == len(data):
        #     data = data.drop(v, axis = 1)
            
    print('Adjusting feature information')
    # Create new rows to be appended
    new_rows = {'Var': ['duration_INSUL_DA', 'duration_ORAL_DAT', 'LDL_c', 'uACR', 'uACR_c'],
         'Group': ['Treatment/Management', 'Treatment/Management', 'Treatment/Management', 'Physiological status', 'Physiological status'],
         'Type': ['Numeric', 'Numeric', 'Categorical', 'Numeric', 'Categorical']
        }   
    new_rows = pd.DataFrame(new_rows)
    # Append the new rows to the var df
    feature_info = pd.concat((feature_info, new_rows), axis = 0)
    #Filter variables
    feature_info = feature_info.loc[feature_info['Var'].isin(data.columns),:]
    
    print('Done')
    
    return(data, feature_info)


def mean_mode_calc(data, feature_info):
    
    """
    Compute mean or mode of the features 
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data to cluster.
    feature_info: dataframe, shape (n_features,)
        Feature info, consisting of at least three columns: 'Var', 'Group' and 'Type'
    
    Returns
    -------
    Dictionary of mean or mode of features
    """
    
    feature_info = feature_info[feature_info['Var'].isin(data.columns)]
    
    # Identify numeric and categorical variables
    num_var = feature_info.loc[feature_info['Type'] == 'Numeric','Var']
    # cat_var = feature_info.loc[feature_info['Type'] == 'Categorical','Var']
    # cat_var = cat_var[~cat_var.isin(['PID','auf_dat','zentrum'])]
    
    # Generate mean and mode of the features
    dic = {}
    for var in num_var:
        missing_mask = np.isnan(data[var])
        mean = np.mean(data[var][~missing_mask])
        dic[var] = mean
        
    # for var in cat_var:
    #     missing_mask = data[var].isna()
    #     mode_val, count_val = mode(data[var][~missing_mask])
    #     if len(data[missing_mask]) > len(data)*0.5:
    #         dic[var] = 'missing'
    #     elif len(mode_val) == 0:
    #         dic[var] = 'missing'
    #     else:
    #         dic[var] = mode_val[0]
        
    return(dic)


def missing_interval_calc(data):
    
    """
    Compute the time interval of missing data from the last observed data 
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data 
    
    Returns
    -------
    Array of shape (n_samples, n_features - 3) showing the time interval 
    """
    
    time = data['auf_dat']
    data = data.drop(['auf_dat','PID','zentrum'], axis=1)
    for var in data.columns:
        
        missing_interval = 0
        col = data[var]
        for idx in data.index:
            if idx == 0:
                col[idx] = 0
            elif pd.isna(data.loc[idx,var]):
                tmp = time[idx] - time[idx - 1]
                tmp = tmp.days
                col[idx] = tmp + missing_interval
                missing_interval = col[idx]
            else:
                tmp = time[idx] - time[idx - 1]
                col[idx] = tmp.days
        data[var] = col    
    
    col_names = list(data.columns) 
    interval_array = data.to_numpy()
    
    return(interval_array, col_names)

def make_dummies(columns, sub_data):
    
    """
    Make one-hot-encoded data preserving universal list of features
    
    Parameters
    ----------
    columns : list
        Complete dummies columns.
    sub_data : array-like, shape (n_visits, n_cat_features)
        Patient data.
        
    Returns
    -------
    One-hot-encoded sub_data 
    """
    
    # One hot encode subset
    encoded_sub_data = pd.get_dummies(sub_data, columns = sub_data.columns)
    
    # Look for missing columns
    missing_columns = list(set(columns) - set(encoded_sub_data.columns))
    
    # Create DataFrame filled with zeros and concatenate to encoded subset
    zeros = pd.DataFrame(np.zeros((len(sub_data), len(missing_columns))), columns=missing_columns)
    encoded_sub_data = pd.concat((encoded_sub_data,zeros), axis = 1)
    final_sub_data = encoded_sub_data.reindex(columns = columns)
    
    return(final_sub_data)


def make_sliding_windows(array, window, stride):
    
    """
    Make multiple sub arrays from a long array using sliding window technique
    
    Parameters
    ----------
    array : numpy array  
        Long array to be split.
    window : int
        Length of the window.
    stride : int
        Number of rows to jump in each step
        
    Returns
    -------
    List of new arrays 
    """
    
    # Calculate the number of sub-arrays you'll get
    num_sub_arrays = (array.shape[0] - window) // stride + 1

    # Initialize a list to store the sub-arrays
    sub_arrays = []

    # Generate sub-arrays using sliding window technique
    for i in range(num_sub_arrays):
        start = i * stride
        end = start + window
        sub_array = array[start:end, :]
        sub_arrays.append(sub_array)

    # Convert the list of sub-arrays to a numpy array
    sub_arrays = np.array(sub_arrays)

    # Print the shape of the resulting sub-arrays
    return(sub_arrays)  # Should be (num_sub_arrays, window_size, num_columns)

    
def preprocessing_data(data, feature_info, mean_mode, result_dir, suffix):
    
    """
    Process input for modeling
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data to cluster.
    feature_info: dataframe, shape (n_features,)
        Feature info, consisting of at least three columns: 'Var', 'Group' and 'Type'
    impute: str
        Imputation approach
    mean_mode: dictionary of length n_features
        Mean and mode of numeric and categorical features, respectively
    
    Returns
    -------
    List of length n_samples of processed data
    """
    
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    print('Starting time: ' + str(datetime.now().time()))
    
    feature_info = feature_info[feature_info['Var'].isin(data.columns)]
    # Identify numeric and categorical variables
    num_var = feature_info.loc[feature_info['Type'] == 'Numeric','Var']
    mean_num = {key: mean_mode[key] for key in num_var}
    cat_var = feature_info.loc[feature_info['Type'] == 'Categorical','Var']
    cat_var = cat_var[~cat_var.isin(['PID','auf_dat','zentrum'])]
    #mode_cat = {key: mean_mode[key] for key in cat_var}
    
	# Sort rows based on patient ID
    data = data.sort_values(by = 'PID')
    
    # One hot encode whole data
    data_cat_filled = data[cat_var].fillna('missing')
    data_cat_encoded = pd.get_dummies(data_cat_filled, columns = cat_var)
    cat_dummies_columns = list(data_cat_encoded.columns)
    
    dataX = []
    pat_inf_arr = []
    
    count = 0
    for i in set(data['PID']):
            
        sub = data[data['PID'] == i].sort_values(by = 'auf_dat')
        sub.reset_index(drop=True, inplace=True)

        if len(sub) < 3: # No. time points >= 3
            continue

        print(str(count) + '.Patient' + str(i), end = " ")

        # Transform date into numeric time vector
        time = sub['auf_dat'] - sub.loc[0,'auf_dat']
        time = time.dt.total_seconds() / (24 * 60 * 60)
        
        time_df = pd.DataFrame({'Time':list(time)})
        patient_info = sub.loc[0,['PID','zentrum']].to_numpy()

        #Extract numeric columns
        sub_num = sub.loc[:,num_var]

        # Iterate over columns to fill up missing values
        sub_num_filled = sub_num.to_numpy()
        missing_mask = np.isnan(sub_num_filled)
        mean_values = np.array(list(mean_num.values()))  # Convert mean values to an array
        if np.any(missing_mask):
            mean_values = np.repeat(mean_values[np.newaxis, :], sub_num_filled.shape[0], axis=0)
            sub_num_filled = np.where(missing_mask, mean_values, sub_num_filled)
        
#         for col_idx in range(sub_num_filled.shape[1]):
            
#             col = sub_num_filled[:, col_idx]
#             missing_mask = np.isnan(col)
#             if np.any(missing_mask):
#                 col[missing_mask] = mean_num[list(mean_num.keys())[col_idx]] # Missing values are replaced with mean value across patients and time points
#                 sub_num_filled[:,col_idx] = col
        
        #Scale to range [0,1]
        #sub_num_filled = (sub_num_filled - np.min(sub_num_filled, 1))/(np.max(sub_num_filled, 1) - np.min(sub_num_filled, 1) + 1e-7)
        # Initialize MinMaxScaler
        #scaler = MinMaxScaler()
        # Scale the DataFrame to range [0, 1]
        #sub_num_filled = scaler.fit_transform(sub_num_filled)
        
        #Extract categorical columns
        sub_cat = sub.loc[:, cat_var]
        # Iterate over columns to fill up missing values
        sub_cat_filled = sub_cat.copy()
        sub_cat_filled.fillna('missing', inplace=True)
#         for col_name in sub_cat.columns:
            
#             if col_name in ['PID','auf_dat','zentrum']:
#                 continue
                
#             col = sub_cat[col_name]
#             missing_mask = col.isna()
#             if np.any(missing_mask):
#                 mode_value = mode_cat[col_name]  # Mode value for the column
#                 sub_cat_filled[col_name].fillna(mode_value, inplace=True)

        # Tranform with one hot encoding
        #sub_cat_filled = pd.get_dummies(sub_cat_filled, columns = sub_cat_filled.columns)
        sub_cat_filled = make_dummies(cat_dummies_columns, sub_cat_filled)
        sub_cat_filled_columns = sub_cat_filled.columns
        sub_cat_filled = sub_cat_filled.to_numpy()

        # Concatenate the arrays column-wise
        sub_processed = np.concatenate((sub_num_filled, sub_cat_filled), axis=1)

        # Identify missing data (for numeric variables only)
        #sub_missing = pd.concat((sub_num, sub_cat), axis=1)
        missing_array = pd.isna(sub_num).astype(int)

        # Compute interval matrix
        #interval_mat, interval_mat_columns = missing_interval_calc(sub)

        # Concatenate the arrays column-wise
        final_array = np.concatenate((time_df, sub_processed, missing_array), axis=1)
        final_array_column = ['Time'] + list(num_var) + list(sub_cat_filled_columns) + [s + '_missing' for s in list(num_var)] #+ [t + '_interval' for t in list(interval_mat_columns)]

        if final_array.shape[0] > 50: 
            final_sub_arrays = make_sliding_windows(final_array, 50, 5) # Split big time series into sliding windows
            for j in range(final_sub_arrays.shape[0]):
                pat_inf_arr.append(patient_info)
                dataX.append(final_sub_arrays[j])
        else:
            dataX.append(final_array)
            pat_inf_arr.append(patient_info)
        
        # Save each batch of 1000 samples
        # if 0 < count < 82001 and count % 1000 == 0: # The iteration # divided by 1000
        #     # if count == 10000:
        #     #     # Reset the list
        #     #     dataX = []
        #     #     pat_inf_arr = []
        #     # else:
        #     # Save data batch
        #     dataX = np.array(dataX, dtype=object)
        #     np.save('/home/phong.nguyen/processed_data/processed_' + str(count) + ".npy", dataX)
        #     pat_inf_arr = np.array(pat_inf_arr, dtype=object)
        #     np.save('/home/phong.nguyen/processed_data/patient_info' + str(count) + '.npy', pat_inf_arr)

        #     # Reset the list
        #     dataX = []
        #     pat_inf_arr = []
            
        count = count + 1

    print('Done.')
    print('Saving results')
    # Save data batch
    # dataX = np.array(dataX, dtype=object)
    # np.save('/home/phong.nguyen/processed_data/processed_' + str(count-1) + ".npy", dataX)
    # pat_inf_arr = np.array(pat_inf_arr, dtype=object)
    # np.save('/home/phong.nguyen/processed_data/patient_info' + str(count-1) + '.npy', pat_inf_arr)
    dataX = np.array(dataX, dtype=object)
    pat_inf_arr = np.array(pat_inf_arr, dtype=object)
    np.save('/home/phong.nguyen/processed_data.npy', dataX)
    np.save('/home/phong.nguyen/data_columns.npy', final_array_column)
    np.save('/home/phong.nguyen/patient_info.npy', pat_inf_arr)
    
    print('Finishing time: ' + str(datetime.now().time()) )
    
    #return(dataX, final_array_column, pat_inf_arr)

def data_padding(data, max_seq_len):
    """Pad the time series to make sequences of equal length
    
    Agrs:
    - data: numpy ndarray of shape (No x Time x Dim)
    - max_seq_len: predefined maximum number of time points
    
    Output:
    - new_data: time series of max_seq_len
    - ori_time: original time info
    """
    
    no = len(data)
    dim = data[0].shape[1]
    
    # Output initialization
    output = np.zeros([no, max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
    time = []
    
    for i in range(len(data)):
        # Extract the time-series data with a certain admissionid
        curr_data = data[i]
        
        # Extract time and assign to the preprocessed data (Excluding ID)
        curr_no = len(curr_data)
        time.append(curr_no)
        
        # Pad data to `max_seq_len`
        if curr_no >= max_seq_len:
            output[i, :, :] = curr_data[:max_seq_len,:]  # Shape: [1, max_seq_len, dim]
        else:
            output[i, :curr_no, :] = curr_data  # Shape: [1, max_seq_len, dim]

    return output, time



def make_case_control(data, target_var, case_val, min_visits):
    """
    Segregate data in cases and controls (for classification task)
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data.
    target_var: str
        Column in data that determines the target variable
    case_val: int or str
        Value in target_val that determines cases
    min_visit: int 
        Minimum number of visits per patient
    
    Returns
    -------
    Two arrays of classes
    """
    
    # Select the relevant columns from data
    visits = data[['PID', 'auf_dat', 'art_n', target_var, 'alter','sex','zentrum']]

    # Group by 'PID' (patient), then calculate the count of visits
    nvis = visits.groupby('PID').size().reset_index(name='n')
    
    # Filter patients by minimum # of visits
    pat = nvis[nvis['n'] >= min_visits]
    visits_fil = visits[visits['PID'].isin(pat['PID'])]
    print('There are ' + str(len(set(pat['PID']))) + ' patients with minimum ' + str(min_visits) + ' visits')
    print('The maximum number of visits is ' + str(nvis['n'].max()) )
    
    # Filter for samples with at least an event
    events = visits_fil[visits_fil[target_var] == case_val]
    visits_evt = visits_fil[visits_fil['PID'].isin(events['PID'])]
     # Filter for samples with no event
    visits_non_evt = visits_fil[~visits_fil['PID'].isin(events['PID'])]
    print('There are ' + str(len(set(visits_evt['PID']))) + ' patients with at least an event')
    print('There are ' + str(len(set(visits_non_evt['PID']))) + ' patients with no event')
    
    print('Identifying cases and matched controls...')
    
    case = []
    ctrl = []
    event_position_case = {}
    target_point_ctrl = {}
    
    for i in set(visits_evt['PID']):
        sub_case = visits_evt[visits_evt['PID'] == i].sort_values(by='auf_dat')
        sub_case.reset_index(drop=True, inplace=True)
        indices = np.where(sub_case[target_var] == case_val)[0]
        if indices.min() == 0:
            continue
        pred_window = sub_case['auf_dat'][indices.min()] - sub_case['auf_dat'][indices.min() - 1] # Compute prediction window
        
        if indices.min() >= 2 and timedelta(14) <= pred_window <= timedelta(365): # The latest event has to be preceded by at least two non-events and prediction window between 14-365 days
            #case.append(i)
            center = sub_case['zentrum'][0]
            sex = sub_case['sex'][0]
            sub_visits = visits_non_evt[visits_non_evt['zentrum'] == center] # Filter for patients in the same center 
            sub_visits = sub_visits[sub_visits['sex'] == sex] # Filter for patients with the same sex
            sub_visits = sub_visits[~sub_visits['PID'].isin(ctrl)] # Get rid of already sampled patients
            if len(sub_visits) == 0:
                continue
            age = sub_case['alter'][0]# Age at first visit
            
            ctrl_for_i = [] # Initialize controls list for patient i
            for j in set(sub_visits['PID']):
                sub_ctrl = sub_visits[sub_visits['PID'] == j].sort_values(by='auf_dat')
                sub_ctrl.reset_index(drop=True, inplace=True)
                age_ctrl = sub_ctrl['alter'][0]
                if age_ctrl <= age - 5 or age_ctrl >= age + 5: # Age at first visit of control is within age range of case's
                    continue
                if sub_ctrl['auf_dat'][0] - sub_case['auf_dat'][0] < timedelta(-60) or sub_ctrl['auf_dat'][0] - sub_case['auf_dat'][0] > timedelta(60): # Control's first visit is two months within the case's 
                    continue
                time_diff = sub_ctrl['auf_dat'] - sub_case['auf_dat'][indices.min()]
                indices2 = np.where(timedelta(-60) <= time_diff <= timedelta(60))[0]
                if len(indices2) == 0: # There should be at least one visit within two months of the case's event
                    continue
                ctrl_for_i.append(j)
                target_point_ctrl[j] = indices2.max()
            if len(ctrl_for_i) > 0:
                case.append(i)
                ctrl = ctrl + ctrl_for_i
                event_position_case[i] = indices.min()
        
    print('The class ' + str(case_val) + ' of the target variable ' + str(target_var) + ' has ' + str(len(case)) + ' eligible cases')
    print('The class ' + str(case_val) + ' of the target variable ' + str(target_var) + ' has ' + str(len(ctrl)) + ' matched eligible controls')
    
    return case, ctrl, event_position_case, target_point_ctrl
    

def make_regression_samples(data, target_var, min_visits, min_pred_window, max_pred_window):
    """
    Identify eligible samples for regression task
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input data.
    target_var: str
        Column in data that determines the target variable
    min_visit: int 
        Minimum number of visits per patient
    min_pred_window: float
        Minimum duration of prediction window
    max_pred_window: float
        Maximum duration of prediction window
    
    Returns
    -------
    An array of selcted samples
    """
    # Select the relevant columns from data
    visits = data[['PID', 'auf_dat', target_var]]

    # Group by 'PID' (patient), then calculate the count of visits
    nvis = visits.groupby('PID').size().reset_index(name='n')
    
    # Filter patients by minimum # of visits
    pat = nvis[nvis['n'] >= min_visits]
    visits_fil = visits[visits['PID'].isin(pat['PID'])]
    print('There are ' + str(len(set(pat['PID']))) + ' patients with minimum ' + str(min_visits) + ' visits')
    print('The maximum number of visits is ' + str(nvis['n'].max()) )
    
    print('Identifying eligible samples...')
    samples = []
    prediction_point = {}
    for i in set(visits_fil['PID']):
        sub = visits_fil[visits_fil['PID'] == i].sort_values(by='auf_dat')
        sub.reset_index(drop=True, inplace=True)
        indices = np.where(~np.isnan(sub[target_var]))[0] # Identify visits where there are records
        if len(indices) == 0 or indices.max() < 2: # Predicted visits have to be preceded by at least 2 visits
            continue
        pred_window = sub['auf_dat'][sub.index.max()] - sub['auf_dat'][sub.index.max() - 1] # Compute prediction window
        if timedelta(min_pred_window) <= pred_window <= timedelta(max_pred_window):
            samples.append(i)
            prediction_point[i] = indices.max()
        else:
            for j in indices:
                pred_window2 = sub['auf_dat'][sub.index[j]] - sub['auf_dat'][sub.index[j-1]] # Prediction window for any other of the recorded visits
                if timedelta(min_pred_window) <= pred_window2 <= timedelta(max_pred_window):
                    samples.append(i)
                    prediction_point[i] = indices.max()
                    break
    
    print('The target variable ' + str(target_var) + ' has ' + str(len(samples)) + ' eligible samples')
    
    return samples, prediction_point

def make_data_bins(data, bin_width = 10):
    
    # Store the original indices
    original_indices = np.argsort(data)

    # Sort the data
    sorted_data = np.sort(data)

    # Determine the bin edges
    min_value = np.min(sorted_data)
    max_value = np.max(sorted_data)
    bin_edges = np.arange(min_value, max_value + bin_width, bin_width)

    # Create an array to hold the transformed data
    transformed_data_sorted = np.empty_like(sorted_data)

    # Process each bin
    for i in range(len(bin_edges) - 1):
        # Determine the current bin range
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Extract the current bin
        bin_mask = (sorted_data >= bin_start) & (sorted_data < bin_end)
        current_bin = sorted_data[bin_mask]

        if len(current_bin) > 0:
            # Calculate the mean of the current bin
            bin_mean = np.mean(current_bin)

            # Replace the original data points in the bin with the bin mean
            transformed_data_sorted[bin_mask] = bin_mean

    # Restore the original order using the stored indices
    transformed_data = np.empty_like(transformed_data_sorted)
    transformed_data[original_indices] = transformed_data_sorted
    
    return transformed_data


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertConfig, RobertaConfig, RobertaModel
import gc
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_validation(last_pos = [0,1,2,3,4]):
    
    result_loss = []
    result_corr = []
    
    for i in last_pos:
        print('Last position: ',i)
        print("...Load data and set up input")
        # Load data
        X = torch.load('/home/phong.nguyen/data_subset_hba1c_long_full10_' + str(i) + '.pth')
        y = torch.load('/home/phong.nguyen/target_subset_hba1c_long_full10_' + str(i) + '.pth')
        
        # Remove the first 4 rows (0s)
        X = X[:,4:,:]
        
        result_loss_i = []
        result_corr_i = []
        
        for seed in range(10):
            print('...Iteration: ',seed)
            # Set the seed for reproducibility and randomly choose 1000 indices without replacement
            torch.manual_seed(seed)
            train_indices = torch.randperm(X.size(0))[:18000]
            torch.manual_seed(seed)
            test_indices = torch.randperm(X.size(0))[18000:]  # all indices except the training ones

            # X_last = X[:,X.size(1)-1,:]
            # X_last_train = X_last[train_indices]
            # y_train = y[train_indices]
            # X_last_test = X_last[test_indices]
            # y_test = y[test_indices]
            
            X_train = X[train_indices]
            X_flat_train = X_train.view(X_train.size(0), -1)
            y_train = y[train_indices]
            X_test = X[test_indices]
            X_flat_test = X_test.view(X_test.size(0), -1)
            y_test = y[test_indices]

            # Instantiation 
            model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 100, seed = 123) 

            # Fitting the model 
            model.fit(X_flat_train, y_train) 

            # xgb_r = xgb.train(params = param, dtrain = train_dmatrix, num_boost_round = 10) 
            pred = model.predict(X_flat_test) 

            def pearson_corrcoef(x, y):
                mean_x = torch.mean(x)
                mean_y = torch.mean(y)
                xm = x - mean_x
                ym = y - mean_y
                r_num = torch.sum(xm * ym)
                r_den = torch.sqrt(torch.sum(xm ** 2)) * torch.sqrt(torch.sum(ym ** 2))
                r = r_num / r_den
                return r

            # MSE Computation 
            loss = MSE(y_test, pred) 
            # Calculate Pearson correlation coefficient
            correlation = pearson_corrcoef(torch.from_numpy(pred), y_test)

            print(f"......Test Loss: {loss.item()}")
            print(f'......Test Pearson Correlation Coefficient: {correlation.item()}')
            
            result_loss_i.append(loss)
            result_corr_i.append(correlation)

        result_loss.append(result_loss_i)
        result_corr.append(result_corr_i)
        
        del X, y
        gc.collect()
    
    return result_loss, result_corr

loss, corr = cross_validation()
np.save("result_xgb_far_loss.npy", loss.cpu().numpy())
np.savetxt("result_xgb_far_loss.csv", loss.cpu().numpy(), delimiter=",")
np.save("result_xgb_far_corr.npy", corr.cpu().numpy())
np.savetxt("result_xgb_far_corr.csv", corr.cpu().numpy(), delimiter=",")

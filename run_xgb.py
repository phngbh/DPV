import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error as MSE
import gc
import numpy as np
import xgboost as xgb
import argparse
from datetime import timedelta, datetime

print('Start running script')
print('Starting time: ' + str(datetime.now().time()))

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for running script')
parser.add_argument('--data', type=str, help='File directory to the processed input data')
parser.add_argument('--target', type=str, help='File directory to the target data')
parser.add_argument('--train_size', type=int, help='Training size')
parser.add_argument('--res_dir', type=str, help='File directory to store the results')
parser.add_argument('--suffix', type=str, help='Name suffix of the results')

args = parser.parse_args()

# Set hyperparameters
data = args.data
target = args.target
train_size = args.train_size
res_dir = args.res_dir
suffix = args.suffix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Load data and set up input")
# Load data
X = torch.load(data)
y = torch.load(target)

# Set the seed for reproducibility and randomly choose 1000 indices without replacement
torch.manual_seed(93)
train_indices = torch.randperm(X.size(0))[:train_size]
torch.manual_seed(93)
test_indices = torch.randperm(X.size(0))[train_size:]

X_last = X[:,48,:]
X_last_train = X_last[train_indices]
y_train = y[train_indices]
X_last_test = X_last[test_indices]
y_test = y[test_indices]

# Instantiation 
model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 100, seed = 123) 

# Fitting the model 
model.fit(X_last_train, y_train) 
  
# xgb_r = xgb.train(params = param, dtrain = train_dmatrix, num_boost_round = 10) 
pred = model.predict(X_last_test) 

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

print(f"Test Loss: {loss.item()}")
print(f'Test Pearson Correlation Coefficient: {correlation.item()}')

torch.save(pred, res_dir + "xgb_pred" + suffix + ".pth")
torch.save(y_test, res_dir + "xgb_y_test" + suffix + ".pth")

print('Finished running script')
print('Finishing time: ' + str(datetime.now().time()))

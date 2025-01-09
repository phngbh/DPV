import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
import warnings
import torch
import os

def make_padding_mask(padded_data, time_seq, device):
    """
    Analyse zero-padded data and output the padding mask  
    
    Parameters
    ----------
    padded_data : array-like
        Zero-padded data, with sample IDs in the first columns.
    time_seq: vector
        Vector of origianl sequence length
    
    Returns
    -------
    padded data without IDs and padding mask
    """
    
    #new_data = padded_data[:,:,1:].to(device)
    pid = padded_data[:,0,0].cpu().numpy().astype(int)
    seq_len = time_seq[pid]
    padding_mask = torch.zeros((padded_data.shape[0], padded_data.shape[1], padded_data.shape[2]-1)).to(device)
    for i in range(padding_mask.shape[0]):
        n = seq_len[i]
        padding_mask[i,:n,:] = 1
    
    return padding_mask

def custom_loss_numeric(predicted, original, padding_mask):
    loss = torch.sum(padding_mask * torch.square(predicted - original)) / torch.sum(padding_mask)
    return loss

def custom_loss_discrete(predicted, original, padding_mask):
    bce_loss = -torch.sum(
        padding_mask * (original * torch.log(predicted + 1e-08) +
                      (1 - original) * torch.log(1 - predicted + 1e-08))
    ) / torch.sum(padding_mask)
    return bce_loss
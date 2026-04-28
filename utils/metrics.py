import numpy as np

def RSE(pred, true, eps=1e-8):
    denom = np.sqrt(np.sum((true - true.mean()) ** 2)) + eps
    return np.sqrt(np.sum((true - pred) ** 2)) / denom

def CORR(pred, true, eps=1e-8):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0)) + eps
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true, eps=1e-6):
    return np.mean(np.abs((pred - true) / (np.abs(true) + eps)))

def MSPE(pred, true, eps=1e-6):
    return np.mean(np.square((pred - true) / (np.abs(true) + eps)))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe
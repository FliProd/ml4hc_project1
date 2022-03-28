import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset


def importData(path_to_data, dataset):
    
    if dataset == 'ptbdb':
        
        df_1 = pd.read_csv(path_to_data + 'ptbdb_normal.csv', header=None)
        df_2 = pd.read_csv(path_to_data + 'ptbdb_abnormal.csv', header=None)
        df = pd.concat([df_1, df_2])

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        Y_test = np.array(df_test[187].values).astype(np.int8)
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

        return (X, Y, X_test, Y_test)
                
    else: 
        
        df_train = pd.read_csv(path_to_data + 'mitbih_train.csv', header=None)
        df_train = df_train.sample(frac=1)
        df_test = pd.read_csv(path_to_data + 'mitbih_test.csv', header=None)

        Y = np.array(df_train[187].values).astype(np.int8)
        X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

        Y_test = np.array(df_test[187].values).astype(np.int8)
        X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

        return (X, Y, X_test, Y_test)


def importDataloader(path_to_data, dataset, batch_size):
     # create features and labels tensor for train set. 
    (X, Y, X_test, Y_test) = importData(path_to_data, dataset)

    X = torch.tensor(X, requires_grad=True).type(torch.float)
    Y = torch.tensor(Y).type(torch.long)
    X_test = torch.tensor(X_test).type(torch.float)
    Y_test = torch.tensor(Y_test).type(torch.long)
   
    # Pytorch train and test sets
    train = TensorDataset(X, Y)
    test = TensorDataset(X_test, Y_test)
    
     # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return (train_loader, test_loader)
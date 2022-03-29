import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_progress_task2():
    base_dir = './reports/figures/'
    progressfiles = ['LSTM_ptbdb_1000_100_2.csv',
                     'Transformer_ptbdb_300_256_3_2_2_128.csv',
                     'Autoencoder_ptbdb_1000_30_2.csv']
    
    df0 = pd.read_csv(base_dir + progressfiles[0])
    x0 = np.array(df0.iloc[:,1])
    y0 = np.array(df0.iloc[:,3])
    
    df1 = pd.read_csv(base_dir + progressfiles[1])
    x1 = np.array(df1.iloc[:,0])[:42] * 2900
    y1 = np.array(df1.iloc[:,1])[:42] * 100


    df2 = pd.read_csv(base_dir + progressfiles[2])
    x2 = np.array(df2.iloc[:,1])
    y2 = np.array(df2.iloc[:,3])
    
    plt.plot(x0, y0, label = "LSTM")
    plt.plot(x1, y1, label = "Transformer")
    plt.plot(x2, y2, label = "Autoencoder")
    plt.legend()
    plt.show()

    
plot_progress()
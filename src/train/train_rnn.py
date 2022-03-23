import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(".")
from src.models.rnn import RNNModel

# includes code from: https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch/notebook

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
        print(Y_test)
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


def trainRNN(dataset):
    path_to_data = "./data/raw/"
    
    input_dim = 187
    if dataset == 'mitbih':
        
        # dataset sepcific hyperparameters
        error = nn.CrossEntropyLoss()
        hidden_dim = 50
        layer_dim = 2
        output_dim = 5 
        
    elif dataset == 'ptbdb':
        dataset_name = 'ptbdb_abnormal.csv'

        # dataset sepcific hyperparameters
        error = nn.BCELoss()
        hidden_dim = 100
        layer_dim = 2
        output_dim = 1
        
    else:
        raise ValueError(dataset, 'is not a valid dataset. Choose mitbih or ptbdb')
    

    # create features and labels tensor for train set. 
    (X, Y, X_test, Y_test) = importData(path_to_data, dataset)

    X = torch.tensor(X, requires_grad=True).type(torch.float)
    Y = torch.tensor(Y).type(torch.long)
    X_test = torch.tensor(X_test).type(torch.float)
    Y_test = torch.tensor(Y_test).type(torch.long)
   
    # Pytorch train and test sets
    train = TensorDataset(X, Y)
    test = TensorDataset(X_test, Y_test)
    
    # batch_size, epoch and iteration
    batch_size = 100
    n_iters = 100000
    num_epochs = int(n_iters/(len(X)/batch_size))
    
     # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    
    # Create RNN & Optimizer
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, output_dim)
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # for visualization
    loss_list = []
    iteration_list = []
    accuracy_list = []
        
    # training loop
    count = 0
    for epoch in range(num_epochs):
        for i, (seq, labels) in enumerate(train_loader):
            
            #ToDo remove this deprecated bariable notations
            seq = Variable(torch.permute(seq, (0, 2, 1)))
            #print(seq.size())
            #print(seq[0])
            labels = Variable(labels)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(seq)

            
            # Calculate softmax and ross entropy loss
            #print('o', outputs.shape)
            #print('o0', outputs[:10])
            #print('la', labels.unsqueeze(1).shape)
            #print('la0', labels[:10])
            loss = error(outputs, labels)
            
            # Calculating gradients
            loss.backward()
            #print('grad', loss.grad)

            
            # Update parameters
            optimizer.step()
            #j = 0
            #for param in model.parameters():
            #    print('param', j, param)
            #    j += 1
            #print('c', count)
            #sys.stdout.write("\033[F")       
            count += 1
            
            if count % 250 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for seq, labels in test_loader:
                    seq = Variable(torch.permute(seq, (0, 2, 1)))
                    
                    # Forward propagation
                    outputs = model(seq)
                    #print('out', outputs.data[:10])
                    # Get predictions from the maximum value
                    if dataset == 'ptbdb':
                        predicted = (outputs.data > 0.5).float()
                    elif dataset == 'mitbih':
                        predicted = torch.max(outputs.data, 1)[1]
                    #print('pred', predicted[:10])
                    #print('lab', labels.unsqueeze(1)[:10])

                    # Total number of labels
                    total += labels.size(0)
                    #print('curr tot', labels.size(0))
                    #print('tot', total)
                    
                    correct += (predicted == labels).sum()
                    #print('curr corr', (predicted == labels.unsqueeze(1)).sum())
                    #print('corr', correct)
                    
                accuracy = 100 * correct / float(total)
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                #print('lol', len(loss_list))
                #print('lol1', loss_list[0])
                #print('i', len(iteration_list))
                #print('i1', iteration_list[0])
                #print('a', len(accuracy_list))
                #print('a0', accuracy_list[len(accuracy_list)-1])
                #print('lod', loss.data.shape)
                #print('lod0', loss.data)
                if count % 1000 == 0:
                    # Print Loss
                    #print('out', outputs.data[:10])
                    #print('pred', predicted[:10])
                    #print('lab', labels.unsqueeze(1)[:10])
                    print('Iteration: {}  Loss: {}  Accuracy: {}%'
                        .format(count, loss.data, accuracy))
    
    visualization(iteration_list, loss_list, accuracy_list)


def visualization(iteration_list, loss_list, accuracy_list):          
    # visualization loss 
    plt.plot(iteration_list,loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("RNN: Loss vs Number of iteration")
    plt.show()

    # visualization accuracy 
    plt.plot(iteration_list,accuracy_list,color = "red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("RNN: Accuracy vs Number of iteration")
    plt.savefig('graph.png')
    plt.show()
    
    
trainRNN('mitbih')
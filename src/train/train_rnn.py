import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(".")
from src.models.rnn import RNNModel
from src.data.load_data import importData
from src.train.util import visualization

# includes code from: https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch/notebook

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
    
    #model summary
    print(model)

    # for visualization
    loss_list = []
    iteration_list = []
    accuracy_list = []
        
    # training loop
    count = 0
    for epoch in range(num_epochs):
        for i, (seq, labels) in enumerate(train_loader):
            
            seq = Variable(torch.permute(seq, (0, 2, 1)))

            labels = Variable(labels)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(seq)

            
            # Calculate softmax and ross entropy loss

            loss = error(outputs, labels)
            
            # Calculating gradients
            loss.backward()

            
            # Update parameters
            optimizer.step()
  
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

                    # Get predictions from the maximum value
                    if dataset == 'ptbdb':
                        predicted = (outputs.data > 0.5).float()
                    elif dataset == 'mitbih':
                        predicted = torch.max(outputs.data, 1)[1]

                    # Total number of labels
                    total += labels.size(0)

                    
                    correct += (predicted == labels).sum()

                    
                accuracy = 100 * correct / float(total)
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

                if count % 1000 == 0:
                    print('Iteration: {}  Loss: {}  Accuracy: {}%'
                        .format(count, loss.data, accuracy))
    
    visualization(iteration_list, loss_list, accuracy_list)


    
    
trainRNN('mitbih')
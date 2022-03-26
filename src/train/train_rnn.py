import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(".")
from src.models.rnn import RNNModel
from src.data.load_data import importData, importDataloader
from src.train.util import progess_visualization

# includes code from: https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch/notebook


# train rnn on specified dataset. If transfer_learning is set to True it will pretrain on dataset on then finetune on ptbdb
def trainRNN(dataset, transfer_learning, fine_tune_dataset):
    path_to_data = "./data/raw/"
    
    input_dim = 187
    hidden_dim = 100
    layer_dim = 2
    error = nn.CrossEntropyLoss()
    
    if dataset == 'mitbih':
        output_dim = 5 
    elif dataset == 'ptbdb':
        output_dim = 2
    else:
        raise ValueError(dataset, 'is not a valid dataset. Choose mitbih or ptbdb')
    
    # batch_size, epoch and iteration
    batch_size = 100
    num_epochs = 100
    
     # get data loaders
    (train_loader, test_loader) = importDataloader(path_to_data=path_to_data,
                                                   dataset=dataset,
                                                   batch_size=batch_size)
    
    # Create RNN & Optimizer
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    (iteration_list, loss_list, accuracy_list) = trainingLoop(model=model, 
                                                              optimizer=optimizer,
                                                              error=error, 
                                                              num_epochs=num_epochs,
                                                              dataset=dataset,
                                                              train_loader=train_loader,
                                                              test_loader=test_loader)
    
    visualization_name = 'train_{}_{}_{}_{}%'.format(dataset, num_epochs, hidden_dim, layer_dim)
    progess_visualization(iteration_list, loss_list, accuracy_list, visualization_name)
    
    
    
    if transfer_learning:
        if fine_tune_dataset == 'mitbih':
            output_dim = 5 
        elif fine_tune_dataset == 'ptbdb':
            output_dim = 2
            
        model.fc = nn.Linear(hidden_dim, output_dim)
        fine_tune_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        (fine_tune_train_loader, fine_tune_test_loader) = importDataloader(path_to_data=path_to_data,
                                                                            dataset=fine_tune_dataset,
                                                                            batch_size=batch_size)
        (iteration_list, loss_list, accuracy_list) =  trainingLoop(model=model, 
                    optimizer=fine_tune_optimizer, 
                    error=error, 
                    num_epochs=num_epochs,
                    dataset=fine_tune_dataset,
                    train_loader=fine_tune_train_loader,
                    test_loader=fine_tune_test_loader)
        
        visualization_name = 'fine_tune_{}_{}_{}_{}%'.format(fine_tune_dataset, num_epochs, hidden_dim, layer_dim)
        progess_visualization(iteration_list, loss_list, accuracy_list, visualization_name)

        




def trainingLoop(model, optimizer, error, num_epochs, dataset, train_loader, test_loader):
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
                    
    return (iteration_list, loss_list, accuracy_list)


    
    
trainRNN(dataset='mitbih', transfer_learning=True, fine_tune_dataset='ptbdb')


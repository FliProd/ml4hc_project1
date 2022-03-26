import torch
import torch.nn as nn
from torch.autograd import Variable


from src.models.rnn import RNNModel
from src.data.load_data import importDataloader
from src.train.util import *


# includes code from: https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch/notebook


# train rnn on specified dataset. If transfer_learning is set to True it will pretrain on dataset on then finetune on ptbdb
def trainRNN(hyperparameters, options):
    
    # load hyperparameters in variables for readability
    dataset_name = options['dataset_name']
    path_to_data = options['path_to_data']
    input_dim = hyperparameters['input_dim']
    hidden_dim = hyperparameters['hidden_dim']
    layer_dim = hyperparameters['layer_dim']
    output_dim = hyperparameters['output_dim'][dataset_name]
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    learning_rate = hyperparameters['learning_rate']
    
    # for storage and logging
    model_name = 'RNN_{}_{}_{}_{}'.format(dataset_name, num_epochs, hidden_dim, layer_dim)

    

     # get data loaders
    (train_loader, test_loader) = importDataloader(path_to_data=path_to_data,
                                                   dataset=dataset_name,
                                                   batch_size=batch_size)
    
    
    # Create RNN & Optimizer
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    
    # train RNN
    try:
        model = torch.load(options['saved_model_path'] + model_name)
        print('loaded model')
        return model
    except IOError:
        print('training from scratch')
        
    (iteration_list, loss_list, accuracy_list) = trainingLoop(model=model, 
                                                              optimizer=optimizer,
                                                              num_epochs=num_epochs,
                                                              train_loader=train_loader,
                                                              test_loader=test_loader)
    
    
    # store model & learning info as .csv
    torch.save(model, options['saved_model_path'] + model_name)
    #storeTrainInfo(iteration_list, loss_list, accuracy_list, model_name)
    
    torch.save(model, options['saved_model_path'] + model_name)

    
    if options['finetune']:
        
        fine_tune_dataset_name = options['fine_tune_dataset_name']
        fine_tune_output_dim = hyperparameters['output_dim'][options['fine_tune_dataset_name']]
        
        # replace last layer to accomodate for new output dimensions
        model.fc = nn.Linear(hidden_dim, fine_tune_output_dim)
        fine_tune_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # load fine tune dataset
        (fine_tune_train_loader, fine_tune_test_loader) = importDataloader(path_to_data=path_to_data,
                                                                            dataset=fine_tune_dataset_name,
                                                                            batch_size=batch_size)
        
        # finetune RNN
        (iteration_list, loss_list, accuracy_list) =  trainingLoop(model=model, 
                                                                    optimizer=fine_tune_optimizer, 
                                                                    num_epochs=num_epochs,
                                                                    train_loader=fine_tune_train_loader,
                                                                    test_loader=fine_tune_test_loader)
        
        torch.save(model, options['saved_model_path'] + model_name)
        #storeTrainInfo(iteration_list, loss_list, accuracy_list, 'fine_tune_' + model_name)
    
    return model

        




def trainingLoop(model, optimizer, num_epochs, train_loader, test_loader):
    #model summary
    print(model)
    
    error = nn.CrossEntropyLoss()
    
    # for progress analysis
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

            # Calculate softmax and ross entropy loss & respective gradients
            loss = error(outputs, labels)
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

                    # stats for progress analysis
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


    


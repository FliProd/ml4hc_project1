import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable


from src.models.lstm import BiLSTMModel
from src.data.load_data import importDataloader




# train lstm on specified dataset.
def trainLSTM(hyperparameters, options):
    
    # load hyperparameters in variables for readability
    dataset_name = options['dataset_name']
    path_to_data = options['path_to_data']
    input_dim = hyperparameters['input_dim']
    hidden_dim = hyperparameters['hidden_dim']
    layer_dim = hyperparameters['layer_dim']
    output_dim = hyperparameters['output_dim'][dataset_name]
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs'][dataset_name]
    learning_rate = hyperparameters['learning_rate']
    
    # for storage and logging
    model_name = 'LSTM_{}_{}_{}_{}'.format(dataset_name, num_epochs, hidden_dim, layer_dim)

    

     # get data loaders
    (train_loader, test_loader) = importDataloader(path_to_data=path_to_data,
                                                   dataset=dataset_name,
                                                   batch_size=batch_size)
    
    
    # Create LSTM & Optimizer
    model = BiLSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    # train LSTM
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
    df = pd.DataFrame(list(zip(iteration_list,loss_list,accuracy_list)))
    df.to_csv(path_or_buf=options['figure_path'] + model_name + '.csv')        
        
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


    


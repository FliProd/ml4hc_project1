from matplotlib import pyplot as plt
import pandas as pd
from src.train.train_rnn import trainRNN
from src.train.train_transformer import trainTransformer
from config.config import config

def get_training_function(model_name):
    if model_name == 'VanillaRNN':
        return trainRNN
    elif model_name == 'Transformer':
        return trainTransformer
    

def progess_visualization(iteration_list, loss_list, accuracy_list, name):          
    
    # visualization loss 
    plt.plot(iteration_list,loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("RNN: Loss vs Number of iteration")
    plt.savefig('./reports/figures/' + name + '_loss.png')
    #plt.show()

    # visualization accuracy 
    plt.plot(iteration_list,accuracy_list,color = "red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("RNN: Accuracy vs Number of iteration")
    plt.savefig(config['figure_path'] + name + '_accuracy.png')
    #plt.show()
    

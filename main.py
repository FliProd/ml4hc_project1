from matplotlib.pyplot import axis
from src.models.hyperparameters import params
from config.config import config
from src.train.util import get_training_function
import keras
from src.models.rnn import RNNModel
from src.data.load_data import importData, importDataloader
import torch
from torch.nn.functional import softmax
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_recall_curve
from matplotlib import pyplot as plt


def main():
    
    model_name = config['model']
    options = config['transfer_learning_options']
    options['path_to_data'] = config['data_dir']
    options['dataset_name'] = config['dataset_name']
    options['saved_model_path'] = config['saved_model_path']
    options['figure_path'] = config['figure_path']
    
    # call training function for model specified in config/config.py
    # pass hyperparameters as named parameters
    model = get_training_function(model_name)(hyperparameters=params[model_name], options=options)
    
    # uncomment to evaluate
    (auroc, auprc) = evaluate(model, model_name)
    

# tests the model on the respective test data set
# prints accuracy and auroc score
# stores RO curve and PR curve in reports/figures
def evaluate(model, model_name):
    dataset = config['dataset_name']
    
    (_, _, X_test, Y_test) = importData(config['data_dir'], config['dataset_name'])

    
    if model_name in config['keras_models']:
        prediction_class_scores = model.predict(X_test)
        name = '{}_{}_{}'.format(model_name, dataset, params[model_name]['num_epochs'])

    elif model_name in config['pytorch_models']:
        X_test = torch.tensor(X_test, requires_grad=False).type(torch.float)
        prediction_class_scores = model(torch.permute(X_test, (0, 2, 1)))
        prediction_class_scores = prediction_class_scores.detach().numpy()
        name = '{}_{}_{}_{}_{}'.format(model_name, dataset, params[model_name]['num_epochs'], params[model_name]['hidden_dim'], params[model_name]['layer_dim'])


    prediction = prediction_class_scores.argmax(axis=1)
    prediction_score = prediction_class_scores[:,1]

    auroc  = []
    auroc_score = 0
    auprc = []
    
    accuracy = accuracy_score(Y_test, prediction)
    if dataset == 'ptbdb':
        auroc = roc_curve(Y_test, prediction_score)
        auroc_score = roc_auc_score(Y_test, prediction_score)
        auprc = precision_recall_curve(Y_test, prediction_score)
    
        print("Accuracy:", accuracy)
        print("AUROC Score:", auroc_score)
        
            
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot(auroc[0], auroc[1])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(name + "AUROC")
        plt.savefig(config['figure_path'] + 'ROC' + name + '.png')
        plt.show()
        
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot(auprc[0], auprc[1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(name + "AUPRC")
        plt.savefig(config['figure_path'] + 'PRC' + name + '.png')
        plt.show()
    
    return auprc, auroc



    
if __name__ == "__main__":
    main()
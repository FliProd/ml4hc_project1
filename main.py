from src.models.hyperparameters import params
from config.config import config
from src.train.util import get_training_function

def main():
    
    model_name = config['model']
    options = config['transfer_learning_options']
    options['path_to_data'] = config['data_dir']
    options['dataset_name'] = config['dataset_name']
    options['saved_model_path'] = config['saved_model_path']
    
    # call training function for model specified in config/config.py
    # pass hyperparameters as named parameters
    get_training_function(model_name)(hyperparameters=params[model_name], options=options)
    
    
    
    
if __name__ == "__main__":
    main()
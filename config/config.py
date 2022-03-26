"""
Training and evaluation settings
"""
config = dict()

"""
Data related settings 
"""
config['data_dir'] = './data/raw/'
config['dataset_name'] = 'mitbih'
config['transfer_learning_options'] = {
    'finetune_dataset_name': 'ptbdb',
    'finetune': False
}

config['sequence_length'] = 187



"""
Model related settings 
Available models: VanillaRNN, Transformer 
"""
config['model'] = 'VanillaRNN'


"""
Path to serialized models & figures & stuff
"""
config['saved_model_path'] = './models/'
config['figure_path'] = './reports/figures/'
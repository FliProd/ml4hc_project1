"""
Training and evaluation settings
"""
config = dict()

"""
Data related settings 
Available datasets: ptbdb, mitbih
"""
config['data_dir'] = './data/raw/'
config['dataset_name'] = 'ptbdb'
config['transfer_learning_options'] = {
    'finetune_dataset_name': 'ptbdb',
    'finetune': False
}

config['sequence_length'] = 187



"""
Model related settings 
Available models: VanillaRNN, Transformer, Autoencoder, BidirectionalLSTM
"""
config['model'] = 'BidirectionalLSTM'

config['pytorch_models'] = ['BidirectionalLSTM', 'Autoencoder', 'VanillaRNN']
config['keras_models'] = ['Transformer']



"""
Path to serialized models & figures & stuff
"""
config['saved_model_path'] = './models/'
config['figure_path'] = './reports/figures/'
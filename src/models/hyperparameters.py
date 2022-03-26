
from config.config import config

params = {
    'VanillaRNN': {
                'input_dim': config['sequence_length'],
                'hidden_dim': 100,
                'layer_dim': 2,
                'output_dim': {'mitbih': 5, 'ptbdb': 2},
                'batch_size': 100,
                'num_epochs': 2,
                'learning_rate': 0.05,
                },
    'Transformer': {
                'n_classes': {'mitbih': 5, 'ptbdb': 2},
                'head_size':256,
                'num_heads':2,
                'ff_dim':2,
                'num_transformer_blocks':2,
                'mlp_units':[256],
                'mlp_dropout':0.4,
                'dropout':0.25,
    },
}

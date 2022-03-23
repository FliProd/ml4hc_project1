import torch
import torch.nn as nn

# modeled after: https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch/notebook

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, num_classes):
        super(RNNModel, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.num_classes = num_classes

    # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = self.init_hidden(batch_size=x.size(0))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        if self.num_classes == 2:
            out = self.sigmoid(out)
        return out
    
    def init_hidden(self, batch_size):
        h0 = torch.rand(self.layer_dim, batch_size, self.hidden_dim)
        return h0
import torch
import torch.nn as nn

#inspired by https://www.youtube.com/watch?v=jGst43P-TJA

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size):
        super(BiLSTMModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_size = hidden_size
        
        # Number of hidden layers
        self.num_layers = num_layers
        
        # BiLSTM 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = torch.rand(self.num_layers*2, x.size(0), self.hidden_size)
        c0 = torch.rand(self.num_layers*2, x.size(0), self.hidden_size)
            
        # One time step
        out, (hidden_state, cell_state) = self.lstm(x, (h0,c0))
        out = self.fc(out[:, -1, :])

        return out
    
    
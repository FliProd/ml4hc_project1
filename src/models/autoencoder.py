import torch
import torch.nn as nn

#inspired by https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size):
        super(Encoder, self).__init__()

     # Number of hidden dimensions
        self.hidden_size = hidden_size
        
        # Number of hidden layers
        self.num_layers = num_layers
        
        # Lstm 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = torch.rand(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.rand(self.num_layers, x.size(0), self.hidden_size)
            
        # One time step
        out, (hidden_state, cell_state) = self.lstm(x, (h0,c0))
        hidden_state = hidden_state[-1, :, :]
        return hidden_state

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size):
        super(Decoder, self).__init__()

     # Number of hidden dimensions
        self.hidden_size = hidden_size
        
        # Number of hidden layers
        self.num_layers = num_layers
        
        # Lstm 
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = torch.rand(self.num_layers, self.hidden_size)
        c0 = torch.rand(self.num_layers, self.hidden_size)
            
        # One time step
        out, (hidden_state, cell_state) = self.lstm(x, (h0,c0))
        out = self.fc(out.squeeze(0))
        return out

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size):
        super(AutoEncoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
    
        # Number of hidden dimensions
        self.hidden_size = hidden_size
        
        # Number of hidden layers
        self.num_layers = num_layers

        self.encoder = Encoder(input_size, hidden_size, num_layers,output_size)
        self.decoder = Decoder(input_size, hidden_size, num_layers,output_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
            
        # One time step
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        """ out = self.fc(decoder_output.squeeze(0)) """
        return decoder_output
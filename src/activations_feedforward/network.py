import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out, hidden_sizes):
        super(NeuralNetwork, self).__init__()
        
        # Create a list of layers as specified
        self.layers = nn.ModuleList([nn.Linear(n_in, hidden_sizes[0])])
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i],hidden_sizes[i+1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], n_out))

    def forward(self, x):
        # Transform from channels x width x height to a single vector
        x = torch.flatten(x, start_dim=1)
        
        for l in self.layers:
            x = F.relu(l(x))
        return x
    
    def activations(self, x):
        # Transform from channels x width x height to a single vector
        x = torch.flatten(x, start_dim=1)
        
        activations = []
        for l in self.layers:
            activations.append(x.detach().to("cpu").numpy())
            x = F.relu(l(x))
        return activations

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, n_out):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode = True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode = True)
            ),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5)
        ])
        self.out = nn.Linear(64, n_out)
        
    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        x = torch.flatten(x, start_dim=1)
        return F.relu(self.out(x))
    
    def activations(self, x):
        activations = [x.detach().to("cpu").numpy()]
        for l in self.layers:
            x = F.relu(l(x))
            activations.append(x.detach().to("cpu").numpy())
        return activations

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from numpy.random import default_rng
from torch import nn

import warnings

rng = default_rng()
device = 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self, size):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(size, 10)
        self.out = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.out(x))
        return x

def load_data(dataset, size, dim, class_to_index):
    filename = 'data/' + dataset 
    
    labels = torch.ones(size, dtype=torch.long).to(device)
    data = torch.ones(size, dim, dtype=torch.float).to(device)
    
    with open(filename) as f:
        for i,line in enumerate(f):
            parts = line.split()
            labels[i] = class_to_index[int(parts[0])]
            for j,n in enumerate(parts[1]):
                data[i,j] = int(n)
            
    return labels,data

def visualize(labels,data,shape,index_to_class,count=3):
    indices = rng.choice(len(labels), size=count, replace=False)
    fig, axs = plt.subplots(1, count)
    fig.suptitle('Dataset examples')

    images = []
    for j in range(count):
        image = torch.reshape(data[indices[j]], shape)
        images.append(axs[j].imshow(image))
        class_index = labels[indices[j]].item()
        axs[j].set_title(index_to_class[class_index])

    plt.show()

def train(data, labels, size, n_classes, model, loss_fn, optimizer):
    pred = model(data)
    loss = loss_fn(pred, labels)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def training_loop(epochs, data, labels, size, n_classes, model, loss_fn, optimizer):
    for t in range(epochs):
        loss = train(data, labels, size, n_classes, model, loss_fn, optimizer)
        if (t % 200 == 0):
            print("Epoch " + str(t))
            print("Loss " + str(loss))

def show_heatmaps(n_classes, model, shape, index_to_class):
    weights_l1 = model.l1.weight.detach()
    weights_out = model.out.weight.detach()
    
    fig, axs = plt.subplots(1, n_classes)
    fig.suptitle('Class heatmaps')
    
    images = []
    for j in range(n_classes):
        out = torch.zeros(n_classes)
        out[j] = 1
        hidden = torch.matmul(torch.transpose(weights_out, 0, 1), out)
        inp = torch.matmul(torch.transpose(weights_l1, 0, 1), hidden)
        image = torch.reshape(inp, shape)
        images.append(axs[j].imshow(image))
        axs[j].set_title(index_to_class[j])

    plt.show()

def main():
    warnings.filterwarnings("ignore")

    available = ["simple"]
    info = {"simple": {
        "size": 9,
        "dim": 25,
        "shape": (5,5),
        "n_classes": 3,
        "class_to_index": {
            1: 0,
            3: 1,
            4: 2
        },
        "index_to_class": [1, 3, 4]
    }}

    print("Select dataset:")
    print("[available: ", end="")
    for i,ds in enumerate(available):
        print(ds, end="")
        if (i != len(available) - 1):
            print(",")
    print("]>", end="")

    ds = input()
    if (ds not in available):
        print("Not a valid dataset")
        return

    size = info[ds]["size"]
    dim = info[ds]["dim"]
    shape = info[ds]["shape"]
    n_classes = info[ds]["n_classes"]
    class_to_index = info[ds]["class_to_index"]
    index_to_class = info[ds]["index_to_class"]
    
    print("Examples of datapoints from dataset:")
    labels, data = load_data(ds, size, dim, class_to_index)
    visualize(labels,data,shape,index_to_class)

    print("Hit enter to continue", end="")
    _ = input()

    model = NeuralNetwork(dim).to(device)
    print("Heatmaps of untrained network")
    show_heatmaps(n_classes, model, shape, index_to_class)

    print("Hit enter to start training", end="")
    _ = input()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    epochs = 2000
    training_loop(epochs, data, labels, size, n_classes, model, loss_fn, optimizer)

    print("Hit enter to continue", end="")
    _ = input()

    print("Heatmaps of trained network")
    show_heatmaps(n_classes, model, shape, index_to_class)


if __name__ == "__main__":
    main()


















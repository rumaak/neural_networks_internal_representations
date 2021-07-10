import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import sys
import logging
import logging.config
import warnings
from copy import deepcopy

from src.dataset_loading import load_MNIST, load_FashionMNIST
from src.activations_feedforward.network import NeuralNetwork
from src.activations_feedforward.training import train_loop, test_loop
from src.print_management import HiddenPrints
from src.activations_feedforward.analysis import get_activations_labels, compute_results
from src.activations_feedforward.embedding import activations_tsne_plot_save, activations_umap_plot_save

def setup_logging(out_dir):
    DEFAULT_LOGGING = {
        'version': 1,
        'formatters': { 
            'standard': {
                'format': '%(asctime)s %(levelname)s: %(message)s',
                'datefmt': '%Y-%m-%d - %H:%M:%S' },
        },
        'handlers': {
            'console':  {'class': 'logging.StreamHandler', 
                         'formatter': "standard", 
                         'level': 'DEBUG', 
                         'stream': sys.stdout},
            'file':     {'class': 'logging.FileHandler', 
                         'formatter': "standard", 
                         'level': 'DEBUG', 
                         'filename': out_dir+'script_log.log','mode': 'w'} 
        },
        'loggers': { 
            __name__:   {'level': 'INFO', 
                         'handlers': ['console', 'file'], 
                         'propagate': False },
        }
    }

    logging.config.dictConfig(DEFAULT_LOGGING)
    log = logging.getLogger(__name__)
    return log

def save_dataframe(df, out_dir, name):
    with open(out_dir+name+'.tex', 'w') as tf:
        tf.write(df.to_latex())

    with open(out_dir+name+'.csv', 'w') as tf:
        tf.write(df.to_csv())

if __name__ == "__main__":
    # Set output directory for logs, figures and tables
    out_dir = '../out/activations_feedforward/'

    # Logger setup
    log = setup_logging(out_dir)

    log.info("Script execution started.")

    # Disable warnings (hides CUDA warnings)
    warnings.filterwarnings("ignore")

    # Use GPU if possible
    if torch.cuda.is_available():
        log.info("Using GPU.")
        device = torch.device('cuda')
    else:
        log.info("Using CPU.")
        device = torch.device('cpu')

    # Data properties
    width, height = 28, 28
    n_classes = 10
    n_examples = 60_000

    # Neural network structure
    hidden_sizes = [512, 256, 128]

    # MNIST dataset
    # -------------

    log.info("Loading the MNIST dataset.")

    # Load data
    training_data, test_data = load_MNIST()
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    # Initialize network
    network = NeuralNetwork(width * height, n_classes, hidden_sizes)
    network.to(device)

    # Save untrained copy of the network
    untrained_state = deepcopy(network.state_dict())
    untrained_network = NeuralNetwork(width * height, n_classes, hidden_sizes)
    untrained_network.to(device).load_state_dict(untrained_state)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-1, momentum=0.9)

    log.info("Neural network training starting.")

    # Train the network until accuracy is greater that 90%
    acc = 0
    while acc < 0.9:
        epochs = 5
        for t in range(epochs):
            with HiddenPrints():
                train_loop(train_dataloader, network, loss_fn, optimizer, device)
                acc = test_loop(test_dataloader, network, loss_fn, device)

        if acc < 0.9:
            log.info("Accuracy too low; restarting training.")

            # Initialize network
            network = NeuralNetwork(width * height, n_classes, hidden_sizes)
            network.to(device)

            # Save untrained copy of the network
            untrained_state = deepcopy(network.state_dict())
            untrained_network = NeuralNetwork(width * height, n_classes, hidden_sizes)
            untrained_network.to(device).load_state_dict(untrained_state)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(network.parameters(), lr=1e-1, momentum=0.9)

    # We won't need the data to be shuffled anymore
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    results = dict()

    # t-SNE, trained network
    log.info("Analysis of t-SNE, trained network.")
    log.info("Computing activations inside the neural network.")
    
    activations, labels = get_activations_labels(
        n_examples,
        width,
        height,
        hidden_sizes,
        train_dataloader,
        network,
        device
    )

    log.info("Computing embeddings.")

    with HiddenPrints():
        embeddings = activations_tsne_plot_save(activations, labels, training_data, out_dir+'mnist_t-sne_trained.png')

    log.info("Computing values of metrics.")

    results[("t-sne", "trained")] = compute_results(embeddings, labels, training_data)

    # t-SNE, untrained network
    log.info("Analysis of t-SNE, untrained network.")
    log.info("Computing activations inside the neural network.")
    
    activations, labels = get_activations_labels(
        n_examples,
        width,
        height,
        hidden_sizes,
        train_dataloader,
        untrained_network,
        device
    )

    log.info("Computing embeddings.")

    with HiddenPrints():
        embeddings = activations_tsne_plot_save(activations, labels, training_data, out_dir+'mnist_t-sne_untrained.png')

    log.info("Computing values of metrics.")

    results[("t-sne", "untrained")] = compute_results(embeddings, labels, training_data)

    # UMAP, trained network
    log.info("Analysis of UMAP, trained network.")
    log.info("Computing activations inside the neural network.")
    
    activations, labels = get_activations_labels(
        n_examples,
        width,
        height,
        hidden_sizes,
        train_dataloader,
        network,
        device
    )

    log.info("Computing embeddings.")

    with HiddenPrints():
        embeddings = activations_umap_plot_save(activations, labels, training_data, out_dir+'mnist_umap_trained.png')

    log.info("Computing values of metrics.")

    results[("umap", "trained")] = compute_results(embeddings, labels, training_data)

    # UMAP, untrained network
    log.info("Analysis of UMAP, untrained network.")
    log.info("Computing activations inside the neural network.")
    
    activations, labels = get_activations_labels(
        n_examples,
        width,
        height,
        hidden_sizes,
        train_dataloader,
        untrained_network,
        device
    )

    log.info("Computing embeddings.")

    with HiddenPrints():
        embeddings = activations_umap_plot_save(activations, labels, training_data, out_dir+'mnist_umap_untrained.png')

    log.info("Computing values of metrics.")

    results[("umap", "untrained")] = compute_results(embeddings, labels, training_data)

    # Analysis
    log.info("Saving the metric data.")

    # t-SNE
    trained_df = pd.DataFrame(results["t-sne", "trained"])
    untrained_df = pd.DataFrame(results["t-sne", "untrained"])

    result_df = pd.DataFrame()
    result_df["ds"] = trained_df["ds"] + " / " + untrained_df["ds"]
    result_df["dd"] = trained_df["dd"] + " / " + untrained_df["dd"]
    result_df["cs"] = trained_df["cs"] + " / " + untrained_df["cs"]
    result_df["cd"] = trained_df["cd"] + " / " + untrained_df["cd"]
    result_df["acc_knn"] = trained_df["acc_knn"] + " / " + untrained_df["acc_knn"]

    row_names = ["layer " + str(i) for i in range(4)]
    result_df = result_df.rename(dict(zip(result_df.index, row_names)))

    save_dataframe(result_df, out_dir, 'mnist_t-sne')

    # UMAP
    trained_df = pd.DataFrame(results["umap", "trained"])
    untrained_df = pd.DataFrame(results["umap", "untrained"])

    result_df = pd.DataFrame()
    result_df["ds"] = trained_df["ds"] + " / " + untrained_df["ds"]
    result_df["dd"] = trained_df["dd"] + " / " + untrained_df["dd"]
    result_df["cs"] = trained_df["cs"] + " / " + untrained_df["cs"]
    result_df["cd"] = trained_df["cd"] + " / " + untrained_df["cd"]
    result_df["acc_knn"] = trained_df["acc_knn"] + " / " + untrained_df["acc_knn"]

    row_names = ["layer " + str(i) for i in range(4)]
    result_df = result_df.rename(dict(zip(result_df.index, row_names)))

    save_dataframe(result_df, out_dir, 'mnist_umap')


    # FashionMNIST dataset
    # --------------------

    log.info("Loading the FashionMNIST dataset.")

    # Load data
    training_data, test_data = load_FashionMNIST()
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    # Initialize network
    network = NeuralNetwork(width * height, n_classes, hidden_sizes)
    network.to(device)

    # Save untrained copy of the network
    untrained_state = deepcopy(network.state_dict())
    untrained_network = NeuralNetwork(width * height, n_classes, hidden_sizes)
    untrained_network.to(device).load_state_dict(untrained_state)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=5e-2, momentum=0.9)

    log.info("Neural network training starting.")

    # Train the network until accuracy is greater that 80%
    acc = 0
    while acc < 0.8:
        epochs = 5
        for t in range(epochs):
            with HiddenPrints():
                train_loop(train_dataloader, network, loss_fn, optimizer, device)
                acc = test_loop(test_dataloader, network, loss_fn, device)

        if acc < 0.8:
            log.info("Accuracy too low; restarting training.")

            # Initialize network
            network = NeuralNetwork(width * height, n_classes, hidden_sizes)
            network.to(device)

            # Save untrained copy of the network
            untrained_state = deepcopy(network.state_dict())
            untrained_network = NeuralNetwork(width * height, n_classes, hidden_sizes)
            untrained_network.to(device).load_state_dict(untrained_state)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(network.parameters(), lr=1e-1, momentum=0.9)

    # We won't need the data to be shuffled anymore
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    results = dict()

    # t-SNE, trained network
    log.info("Analysis of t-SNE, trained network.")
    log.info("Computing activations inside the neural network.")
    
    activations, labels = get_activations_labels(
        n_examples,
        width,
        height,
        hidden_sizes,
        train_dataloader,
        network,
        device
    )

    log.info("Computing embeddings.")

    with HiddenPrints():
        embeddings = activations_tsne_plot_save(activations, labels, training_data, out_dir+'fmnist_t-sne_trained.png')

    log.info("Computing values of metrics.")

    results[("t-sne", "trained")] = compute_results(embeddings, labels, training_data)

    # t-SNE, untrained network
    log.info("Analysis of t-SNE, untrained network.")
    log.info("Computing activations inside the neural network.")
    
    activations, labels = get_activations_labels(
        n_examples,
        width,
        height,
        hidden_sizes,
        train_dataloader,
        untrained_network,
        device
    )

    log.info("Computing embeddings.")

    with HiddenPrints():
        embeddings = activations_tsne_plot_save(activations, labels, training_data, out_dir+'fmnist_t-sne_untrained.png')

    log.info("Computing values of metrics.")

    results[("t-sne", "untrained")] = compute_results(embeddings, labels, training_data)

    # UMAP, trained network
    log.info("Analysis of UMAP, trained network.")
    log.info("Computing activations inside the neural network.")
    
    activations, labels = get_activations_labels(
        n_examples,
        width,
        height,
        hidden_sizes,
        train_dataloader,
        network,
        device
    )

    log.info("Computing embeddings.")

    with HiddenPrints():
        embeddings = activations_umap_plot_save(activations, labels, training_data, out_dir+'fmnist_umap_trained.png')

    log.info("Computing values of metrics.")

    results[("umap", "trained")] = compute_results(embeddings, labels, training_data)

    # UMAP, untrained network
    log.info("Analysis of UMAP, untrained network.")
    log.info("Computing activations inside the neural network.")
    
    activations, labels = get_activations_labels(
        n_examples,
        width,
        height,
        hidden_sizes,
        train_dataloader,
        untrained_network,
        device
    )

    log.info("Computing embeddings.")

    with HiddenPrints():
        embeddings = activations_umap_plot_save(activations, labels, training_data, out_dir+'fmnist_umap_untrained.png')

    log.info("Computing values of metrics.")

    results[("umap", "untrained")] = compute_results(embeddings, labels, training_data)

    # Analysis
    log.info("Saving the metric data.")

    # t-SNE
    trained_df = pd.DataFrame(results["t-sne", "trained"])
    untrained_df = pd.DataFrame(results["t-sne", "untrained"])

    result_df = pd.DataFrame()
    result_df["ds"] = trained_df["ds"] + " / " + untrained_df["ds"]
    result_df["dd"] = trained_df["dd"] + " / " + untrained_df["dd"]
    result_df["cs"] = trained_df["cs"] + " / " + untrained_df["cs"]
    result_df["cd"] = trained_df["cd"] + " / " + untrained_df["cd"]
    result_df["acc_knn"] = trained_df["acc_knn"] + " / " + untrained_df["acc_knn"]

    row_names = ["layer " + str(i) for i in range(4)]
    result_df = result_df.rename(dict(zip(result_df.index, row_names)))

    save_dataframe(result_df, out_dir, 'fmnist_t-sne')

    # UMAP
    trained_df = pd.DataFrame(results["umap", "trained"])
    untrained_df = pd.DataFrame(results["umap", "untrained"])

    result_df = pd.DataFrame()
    result_df["ds"] = trained_df["ds"] + " / " + untrained_df["ds"]
    result_df["dd"] = trained_df["dd"] + " / " + untrained_df["dd"]
    result_df["cs"] = trained_df["cs"] + " / " + untrained_df["cs"]
    result_df["cd"] = trained_df["cd"] + " / " + untrained_df["cd"]
    result_df["acc_knn"] = trained_df["acc_knn"] + " / " + untrained_df["acc_knn"]

    row_names = ["layer " + str(i) for i in range(4)]
    result_df = result_df.rename(dict(zip(result_df.index, row_names)))

    save_dataframe(result_df, out_dir, 'fmnist_umap')
















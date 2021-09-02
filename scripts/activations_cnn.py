import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt

import sys
import logging
import logging.config
import warnings
from copy import deepcopy

from src.dataset_loading import load_MNIST, load_FashionMNIST
from src.activations_cnn.network import NeuralNetwork
from src.activations_cnn.training import train_loop, test_loop
from src.print_management import HiddenPrints
from src.activations_cnn.analysis import get_activations_labels, single_embedding_metrics, analyze_outliers, analyze_filters
from src.activations_cnn.embedding import activations_tsne_plot_save, activations_umap_plot_save

def setup_logging(out_dir):
    """Initialize logging"""
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

def update_results(results, embeddings, dataset_name, network_type, labels, training_data):
    """Computes metrics for a set of embeddings and save them into `results`"""
    e0, e1, e2, e3 = embeddings 
    
    results[0][(dataset_name, network_type)] = [(single_embedding_metrics(e0, labels, training_data))]
    
    results[1][(dataset_name, network_type)] = []
    for e in e1:
        results[1][(dataset_name, network_type)].append(single_embedding_metrics(e, labels, training_data))
        
    results[2][(dataset_name, network_type)] = []
    for e in e2:
        results[2][(dataset_name, network_type)].append(single_embedding_metrics(e, labels, training_data))
        
    results[3][(dataset_name, network_type)] = [(single_embedding_metrics(e3, labels, training_data))]

def save_dataframe(df, out_dir, name):
    """Save dataframe in .csv and .tex format"""
    with open(out_dir+name+'.tex', 'w') as tf:
        tf.write(df.to_latex())

    with open(out_dir+name+'.csv', 'w') as tf:
        tf.write(df.to_csv())

def save(results, output_dir):
    """Save results to filesystem"""
    for layer in range(4):
        layer_result = results[layer]
        
        for method in ["t-sne", "umap"]:
            trained_df = pd.DataFrame(layer_result[method, "trained"])
            untrained_df = pd.DataFrame(layer_result[method, "untrained"])

            result_df = pd.DataFrame()
            result_df["ds"] = trained_df["ds"] + " / " + untrained_df["ds"]
            result_df["dd"] = trained_df["dd"] + " / " + untrained_df["dd"]
            result_df["cs"] = trained_df["cs"] + " / " + untrained_df["cs"]
            result_df["cd"] = trained_df["cd"] + " / " + untrained_df["cd"]
            result_df["acc_knn"] = trained_df["acc_knn"] + " / " + untrained_df["acc_knn"]

            row_names = ["filter " + str(i+1) for i in range(len(trained_df))]
            result_df = result_df.rename(dict(zip(result_df.index, row_names)))

            output_dir_path = output_dir + "/" + method + "/"
            save_dataframe(result_df, output_dir_path, "table_l" + str(layer))

def metrics_plots(results, output_dir, method, name):
    """Creates and saves plots of average values of metrics"""
    metrics = ["ds", "dd", "cs", "cd", "acc_knn"]
    
    trained_vals = dict()
    untrained_vals = dict()
    for m in metrics:
        trained_vals[m] = []
        untrained_vals[m] = []
        
    for layer in results:
        layer_result = results[layer]
        trained_df = pd.DataFrame(layer_result[method, "trained"]).astype(float)
        untrained_df = pd.DataFrame(layer_result[method, "untrained"]).astype(float)
    
        for m in metrics:
            trained_vals[m].append(np.average(trained_df[m]))
            untrained_vals[m].append(np.average(untrained_df[m]))
            
    for m in metrics:
        fig, axes = plt.subplots()
        axes.plot(trained_vals[m], marker=".", label="trained")
        axes.plot(untrained_vals[m], marker=".", label="untrained")
        axes.set(
            xlabel="network depth",
            ylabel=m,
        )
        axes.legend()
        plt.savefig(output_dir + "/" + method + "/" + name + "_" + m + ".png")
        plt.close()

if __name__ == "__main__":
    # Set output directory for logs, figures and tables
    out_dir = '../out/activations_cnn/'

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

    # Initialize numpy rng
    rng = default_rng()

    # Data properties
    width, height = 28, 28
    n_classes = 10
    n_examples = 60_000

    # MNIST dataset
    # -------------

    log.info("Loading the MNIST dataset.")

    # Load data
    training_data, test_data = load_MNIST()
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    # Initialize network
    network = NeuralNetwork(n_classes)
    network.to(device)

    # Save untrained copy of the network
    untrained_state = deepcopy(network.state_dict())
    untrained_network = NeuralNetwork(n_classes)
    untrained_network.to(device).load_state_dict(untrained_state)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, momentum=0.9)

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
            network = NeuralNetwork(n_classes)
            network.to(device)

            # Save untrained copy of the network
            untrained_state = deepcopy(network.state_dict())
            untrained_network = NeuralNetwork(n_classes)
            untrained_network.to(device).load_state_dict(untrained_state)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, momentum=0.9)


    # We won't need the data to be shuffled anymore
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    results = {
        0: dict(),
        1: dict(),
        2: dict(),
        3: dict()
    }

    # t-SNE, trained network
    log.info("Analysis of t-SNE, trained network.")
    log.info("Computing activations inside the neural network.")

    # Compute activations inside the trained neural network
    activations_zero, activations_first, activations_second, activations_third, labels = get_activations_labels(
        n_examples,
        width,
        height,
        train_dataloader,
        network,
        device
    )

    # take smaller subset of data
    labels = labels[:5_000]
    activations_zero = activations_zero[:5000]

    for i in range(4):
        activations_first[i] = activations_first[i][:5000]
        
    for i in range(16):
        activations_second[i] = activations_second[i][:5000]

    activations_third = activations_third[:5000]

    log.info("Computing embeddings.")

    # Compute low-dimensional embeddings from the activations
    with HiddenPrints():
        embeddings = activations_tsne_plot_save(
            activations_zero,
            activations_first,
            activations_second,
            activations_third,
            labels,
            training_data,
            out_dir + "mnist/t-sne/trained/plot"
        )

    log.info("Computing values of metrics.")

    # Compute metric values over the embeddings
    update_results(results, embeddings, "t-sne", "trained", labels, training_data)

    # Analysis of the third layer
    _, _, _, e3 = embeddings 
    analyze_outliers(training_data, e3, labels, out_dir + "mnist/t-sne/outliers/")

    # Analysis of filters
    analyze_filters(
        training_data,
        out_dir + "mnist/t-sne/filter_outputs/",
        network,
        rng,
        device
    )

    # UMAP, trained network
    log.info("Analysis of UMAP, trained network.")
    log.info("Computing embeddings.")

    # Compute low-dimensional embeddings from the activations
    with HiddenPrints():
        embeddings = activations_umap_plot_save(
            activations_zero,
            activations_first,
            activations_second,
            activations_third,
            labels,
            training_data,
            out_dir + "mnist/umap/trained/plot"
        )

    log.info("Computing values of metrics.")

    # Compute metric values over the embeddings
    update_results(results, embeddings, "umap", "trained", labels, training_data)

    # t-SNE, untrained network
    log.info("Analysis of t-SNE, untrained network.")
    log.info("Computing activations inside the untrained neural network.")

    # Compute activations inside the untrained neural network
    activations_zero, activations_first, activations_second, activations_third, labels = get_activations_labels(
        n_examples,
        width,
        height,
        train_dataloader,
        untrained_network,
        device
    )

    # take smaller subset of data
    labels = labels[:5_000]
    activations_zero = activations_zero[:5000]

    for i in range(4):
        activations_first[i] = activations_first[i][:5000]
        
    for i in range(16):
        activations_second[i] = activations_second[i][:5000]

    activations_third = activations_third[:5000]

    log.info("Computing embeddings.")

    # Compute low-dimensional embeddings from the activations
    with HiddenPrints():
        embeddings = activations_tsne_plot_save(
            activations_zero,
            activations_first,
            activations_second,
            activations_third,
            labels,
            training_data,
            out_dir + "mnist/t-sne/untrained/plot"
        )

    log.info("Computing values of metrics.")

    # Compute metric values over the embeddings
    update_results(results, embeddings, "t-sne", "untrained", labels, training_data)

    # UMAP, untrained network
    log.info("Analysis of UMAP, untrained network.")
    log.info("Computing embeddings.")

    # Compute low-dimensional embeddings from the activations
    with HiddenPrints():
        embeddings = activations_umap_plot_save(
            activations_zero,
            activations_first,
            activations_second,
            activations_third,
            labels,
            training_data,
            out_dir + "mnist/umap/untrained/plot"
        )

    log.info("Computing values of metrics.")

    # Compute metric values over the embeddings
    update_results(results, embeddings, "umap", "untrained", labels, training_data)

    # Save the results (computed metrics)
    save(results, out_dir + "mnist/")
    for method in ["t-sne", "umap"]:
        metrics_plots(results, out_dir + "mnist/", method, "plot_metric")

    # # FashionMNIST dataset
    # # --------------------

    log.info("Loading the FashionMNIST dataset.")

    # Load data
    training_data, test_data = load_FashionMNIST()
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    # Initialize network
    network = NeuralNetwork(n_classes)
    network.to(device)

    # Save untrained copy of the network
    untrained_state = deepcopy(network.state_dict())
    untrained_network = NeuralNetwork(n_classes)
    untrained_network.to(device).load_state_dict(untrained_state)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=3e-2, momentum=0.9)

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
            network = NeuralNetwork(n_classes)
            network.to(device)

            # Save untrained copy of the network
            untrained_state = deepcopy(network.state_dict())
            untrained_network = NeuralNetwork(n_classes)
            untrained_network.to(device).load_state_dict(untrained_state)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(network.parameters(), lr=3e-2, momentum=0.9)

    # We won't need the data to be shuffled anymore
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    results = {
        0: dict(),
        1: dict(),
        2: dict(),
        3: dict()
    }

    # t-SNE, trained network
    log.info("Analysis of t-SNE, trained network.")
    log.info("Computing activations inside the neural network.")

    # Compute activations inside the trained neural network
    activations_zero, activations_first, activations_second, activations_third, labels = get_activations_labels(
        n_examples,
        width,
        height,
        train_dataloader,
        network,
        device
    )

    # take smaller subset of data
    labels = labels[:5_000]
    activations_zero = activations_zero[:5000]

    for i in range(4):
        activations_first[i] = activations_first[i][:5000]
        
    for i in range(16):
        activations_second[i] = activations_second[i][:5000]

    activations_third = activations_third[:5000]

    log.info("Computing embeddings.")

    # Compute low-dimensional embeddings from the activations
    with HiddenPrints():
        embeddings = activations_tsne_plot_save(
            activations_zero,
            activations_first,
            activations_second,
            activations_third,
            labels,
            training_data,
            out_dir + "fmnist/t-sne/trained/plot"
        )

    log.info("Computing values of metrics.")

    # Compute metric values over the embeddings
    update_results(results, embeddings, "t-sne", "trained", labels, training_data)

    # Analysis of the third layer
    _, _, _, e3 = embeddings 
    analyze_outliers(training_data, e3, labels, out_dir + "fmnist/t-sne/outliers/")

    # Analysis of filters
    analyze_filters(
        training_data,
        out_dir + "fmnist/t-sne/filter_outputs/",
        network,
        rng,
        device
    )

    # UMAP, trained network
    log.info("Analysis of UMAP, trained network.")
    log.info("Computing embeddings.")

    # Compute low-dimensional embeddings from the activations
    with HiddenPrints():
        embeddings = activations_umap_plot_save(
            activations_zero,
            activations_first,
            activations_second,
            activations_third,
            labels,
            training_data,
            out_dir + "fmnist/umap/trained/plot"
        )

    log.info("Computing values of metrics.")

    # Compute metric values over the embeddings
    update_results(results, embeddings, "umap", "trained", labels, training_data)

    # t-SNE, untrained network
    log.info("Analysis of t-SNE, untrained network.")
    log.info("Computing activations inside the untrained neural network.")

    # Compute activations inside the untrained neural network
    activations_zero, activations_first, activations_second, activations_third, labels = get_activations_labels(
        n_examples,
        width,
        height,
        train_dataloader,
        untrained_network,
        device
    )

    # take smaller subset of data
    labels = labels[:5_000]
    activations_zero = activations_zero[:5000]

    for i in range(4):
        activations_first[i] = activations_first[i][:5000]
        
    for i in range(16):
        activations_second[i] = activations_second[i][:5000]

    activations_third = activations_third[:5000]

    log.info("Computing embeddings.")

    # Compute low-dimensional embeddings from the activations
    with HiddenPrints():
        embeddings = activations_tsne_plot_save(
            activations_zero,
            activations_first,
            activations_second,
            activations_third,
            labels,
            training_data,
            out_dir + "fmnist/t-sne/untrained/plot"
        )

    log.info("Computing values of metrics.")

    # Compute metric values over the embeddings
    update_results(results, embeddings, "t-sne", "untrained", labels, training_data)

    # UMAP, untrained network
    log.info("Analysis of UMAP, untrained network.")
    log.info("Computing embeddings.")

    # Compute low-dimensional embeddings from the activations
    with HiddenPrints():
        embeddings = activations_umap_plot_save(
            activations_zero,
            activations_first,
            activations_second,
            activations_third,
            labels,
            training_data,
            out_dir + "fmnist/umap/untrained/plot"
        )

    log.info("Computing values of metrics.")

    # Compute metric values over the embeddings
    update_results(results, embeddings, "umap", "untrained", labels, training_data)

    # Save the results (computed metrics)
    save(results, out_dir + "fmnist/")
    for method in ["t-sne", "umap"]:
        metrics_plots(results, out_dir + "fmnist/", method, "plot_metric")



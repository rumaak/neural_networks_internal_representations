import torch
import numpy as np

from .metrics import same_diff_average, dist_to_centroid, dist_between_centroids, knn_evaluate

def activations_loop(dataloader, model, activations, labels, device):
    """Compute activations inside the neural network for all examples."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    index = 0
    with torch.no_grad():
        for X,y in dataloader:
            # Use GPU if possible
            X,y = X.to(device), y.to(device)
            
            batch_size = X.shape[0]
            acts = model.activations(X)
            
            for i,a in enumerate(acts):
                activations[i][index:(index+batch_size)] = a
            labels[index:(index+batch_size)] = y.detach().to("cpu").numpy()
            
            index += batch_size

def compute_results(embeddings, labels, training_data):
    """Apply metrics to supplied embeddings."""
    metrics = []
    for emb in embeddings:
        emb = np.array(emb)
        metric_values = dict()

        same, diff, average = same_diff_average(emb, labels)

        metric_values["ds"] = f"{(same / average):.4f}"
        metric_values["dd"] = f"{(diff / average):.4f}"
        
        to_centroid = dist_to_centroid(
            emb,
            labels,
            training_data.class_to_idx.values()
        ) / average
        metric_values["cs"] = f"{to_centroid:.4f}"
        
        between_centroids = dist_between_centroids(
            emb, 
            labels, 
            training_data.class_to_idx.values()
        ) / average 
        
        metric_values["cd"] = f"{between_centroids:.4f}"
        metric_values["acc_knn"] = f"{knn_evaluate(emb, labels):.4f}"
        
        metrics.append(metric_values)
        
    return metrics

def get_activations_labels(n_examples, width, height, hidden_sizes, train_dataloader, network, device):
    """Get activations inside the neural network for all examples."""
    activations = []
    activations.append(np.zeros((n_examples, width * height)))
    for hs in hidden_sizes:
        activations.append(np.zeros((n_examples, hs)))

    labels = np.zeros((60_000))
    
    activations_loop(train_dataloader, network, activations, labels, device)
    labels = labels.astype(np.int32)
    
    # labels = labels[:5_000]
    # activations_small = []
    # 
    # for a in activations:
    #     activations_small.append(a[:5_000])
    # activations = activations_small
    
    return activations, labels


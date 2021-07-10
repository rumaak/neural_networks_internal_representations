import umap
from openTSNE import TSNE

import matplotlib.pyplot as plt
import numpy as np

def activations_tsne_plot(activations, labels, ds):
    """Compute embeddings using t-SNE and plot them."""
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=False,
    )
    fig, axes = plt.subplots(nrows=1, ncols=len(activations), figsize=(25,5))

    embs = []
    for idx, acts in enumerate(activations):
        print("Learning embeddings for layer " + str(idx) + "...")
        embeddings = tsne.fit(acts)

        for i,actual_label in enumerate(ds.classes):
            indices = np.argwhere(labels == i)
            indices = np.squeeze(indices)
            
            axes[idx].scatter(embeddings[indices,0],embeddings[indices,1],label=actual_label,s=2)
            axes[idx].legend()
            axes[idx].set_title("Activations in layer " + str(idx))
            
        embs.append(embeddings)

    fig.tight_layout()
    return embs

def activations_tsne_plot_save(activations, labels, ds, filename):
    """Compute embeddings using t-SNE and save their plots."""
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=False,
    )
    fig, axes = plt.subplots(nrows=1, ncols=len(activations), figsize=(25,5))

    embs = []
    for idx, acts in enumerate(activations):
        print("Learning embeddings for layer " + str(idx) + "...")
        embeddings = tsne.fit(acts)

        for i,actual_label in enumerate(ds.classes):
            indices = np.argwhere(labels == i)
            indices = np.squeeze(indices)
            
            axes[idx].scatter(embeddings[indices,0],embeddings[indices,1],label=actual_label,s=2)
            axes[idx].legend()
            axes[idx].set_title("Activations in layer " + str(idx))
            
        embs.append(embeddings)

    fig.tight_layout()
    plt.savefig(filename)

    return embs

def activations_umap_plot(activations, labels, ds):
    """Compute embeddings using UMAP and plot them."""
    reducer = umap.UMAP()
    fig, axes = plt.subplots(nrows=1, ncols=len(activations), figsize=(25,5))
            
    embs = []
    for idx, acts in enumerate(activations):
        print("Learning embeddings for layer " + str(idx) + "...")
        embeddings = reducer.fit_transform(acts)

        for i,actual_label in enumerate(ds.classes):
            indices = np.argwhere(labels == i)
            indices = np.squeeze(indices)
            
            axes[idx].scatter(embeddings[indices,0],embeddings[indices,1],label=actual_label,s=2)
            axes[idx].legend()
            axes[idx].set_title("Activations in layer " + str(idx))
            
        embs.append(embeddings)

    fig.tight_layout()
    return embs

def activations_umap_plot_save(activations, labels, ds, filename):
    """Compute embeddings using UMAP and save their plots."""
    reducer = umap.UMAP()
    fig, axes = plt.subplots(nrows=1, ncols=len(activations), figsize=(25,5))
            
    embs = []
    for idx, acts in enumerate(activations):
        print("Learning embeddings for layer " + str(idx) + "...")
        embeddings = reducer.fit_transform(acts)

        for i,actual_label in enumerate(ds.classes):
            indices = np.argwhere(labels == i)
            indices = np.squeeze(indices)
            
            axes[idx].scatter(embeddings[indices,0],embeddings[indices,1],label=actual_label,s=2)
            axes[idx].legend()
            axes[idx].set_title("Activations in layer " + str(idx))
            
        embs.append(embeddings)

    fig.tight_layout()
    plt.savefig(filename)

    return embs

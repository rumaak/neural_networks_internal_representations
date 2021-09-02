import umap
from openTSNE import TSNE

import matplotlib.pyplot as plt
import numpy as np

def activations_tsne_plot_save(activations_zero, activations_first,
    activations_second, activations_third, labels, ds, filename):
    """Compute embeddings using t-SNE and save their plots."""
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=False,
    )
    
    # embeddings_zero
    print("Learning embeddings for original dataset")
    embeddings_zero = tsne.fit(activations_zero)
    
    fig, axes = plt.subplots(figsize=(12,8))
    for i,actual_label in enumerate(ds.classes):
        indices = np.argwhere(labels == i)
        indices = np.squeeze(indices)
        
        axes.scatter(embeddings_zero[indices,0],embeddings_zero[indices,1],label=actual_label,s=12)
        axes.legend(markerscale=3, fontsize=12)

    plt.savefig(filename + "_l0.png")
    plt.close()
    
    # embeddings_first
    embeddings_first = []
    print("Learning embeddings for first layer")
    for j,acts in enumerate(activations_first):
        embedding = tsne.fit(acts)
        embeddings_first.append(embedding)
    
        fig, axes = plt.subplots(figsize=(12,8))
        for i,actual_label in enumerate(ds.classes):
            indices = np.argwhere(labels == i)
            indices = np.squeeze(indices)

            axes.scatter(embedding[indices,0],embedding[indices,1],label=actual_label,s=12)
            axes.legend(markerscale=3, fontsize=12)

        plt.savefig(filename + "_l1_f" + str(j) + ".png")
        plt.close()
        
    # embeddings_second
    embeddings_second = []
    print("Learning embeddings for second layer")
    for j,acts in enumerate(activations_second):
        embedding = tsne.fit(acts)
        embeddings_second.append(embedding)
    
        fig, axes = plt.subplots(figsize=(12,8))
        for i,actual_label in enumerate(ds.classes):
            indices = np.argwhere(labels == i)
            indices = np.squeeze(indices)

            axes.scatter(embedding[indices,0],embedding[indices,1],label=actual_label,s=12)
            axes.legend(markerscale=3, fontsize=12)

        plt.savefig(filename + "_l2_f" + str(j) + ".png")
        plt.close()
    
    # embeddings_third
    print("Learning embeddings for third layer")
    embeddings_third = tsne.fit(activations_third)
    
    fig, axes = plt.subplots(figsize=(12,8))
    for i,actual_label in enumerate(ds.classes):
        indices = np.argwhere(labels == i)
        indices = np.squeeze(indices)
        
        axes.scatter(embeddings_third[indices,0],embeddings_third[indices,1],label=actual_label,s=12)
        axes.legend(markerscale=3, fontsize=12)
            
    plt.savefig(filename + "_l3.png")
    plt.close()
    
    return embeddings_zero, embeddings_first, embeddings_second, embeddings_third

def activations_umap_plot_save(activations_zero, activations_first,
    activations_second, activations_third, labels, ds, filename):
    """Compute embeddings using UMAP and save their plots."""
    reducer = umap.UMAP()
    
    # embeddings_zero
    print("Learning embeddings for original dataset")
    embeddings_zero = reducer.fit_transform(activations_zero)
    
    fig, axes = plt.subplots(figsize=(12,8))
    for i,actual_label in enumerate(ds.classes):
        indices = np.argwhere(labels == i)
        indices = np.squeeze(indices)
        
        axes.scatter(embeddings_zero[indices,0],embeddings_zero[indices,1],label=actual_label,s=12)
        axes.legend(markerscale=3, fontsize=12)

    plt.savefig(filename + "_l0.png")
    plt.close()
    
    # embeddings_first
    embeddings_first = []
    print("Learning embeddings for first layer")
    for j,acts in enumerate(activations_first):
        embedding = reducer.fit_transform(acts)
        embeddings_first.append(embedding)
    
        fig, axes = plt.subplots(figsize=(12,8))
        for i,actual_label in enumerate(ds.classes):
            indices = np.argwhere(labels == i)
            indices = np.squeeze(indices)

            axes.scatter(embedding[indices,0],embedding[indices,1],label=actual_label,s=12)
            axes.legend(markerscale=3, fontsize=12)

        plt.savefig(filename + "_l1_f" + str(j) + ".png")
        plt.close()
        
    # embeddings_second
    embeddings_second = []
    print("Learning embeddings for second layer")
    for j,acts in enumerate(activations_second):
        embedding = reducer.fit_transform(acts)
        embeddings_second.append(embedding)
    
        fig, axes = plt.subplots(figsize=(12,8))
        for i,actual_label in enumerate(ds.classes):
            indices = np.argwhere(labels == i)
            indices = np.squeeze(indices)

            axes.scatter(embedding[indices,0],embedding[indices,1],label=actual_label,s=12)
            axes.legend(markerscale=3, fontsize=12)

        plt.savefig(filename + "_l2_f" + str(j) + ".png")
        plt.close()
    
    # embeddings_third
    print("Learning embeddings for third layer")
    embeddings_third = reducer.fit_transform(activations_third)
    
    fig, axes = plt.subplots(figsize=(12,8))
    for i,actual_label in enumerate(ds.classes):
        indices = np.argwhere(labels == i)
        indices = np.squeeze(indices)
        
        axes.scatter(embeddings_third[indices,0],embeddings_third[indices,1],label=actual_label,s=12)
        axes.legend(markerscale=3, fontsize=12)
            
    plt.savefig(filename + "_l3.png")
    plt.close()
    
    return embeddings_zero, embeddings_first, embeddings_second, embeddings_third

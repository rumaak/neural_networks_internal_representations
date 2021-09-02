import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from src.activations_cnn.metrics import same_diff_average, dist_to_centroid, dist_between_centroids, knn_evaluate

def single_embedding_metrics(emb, labels, ds):
    """Apply metrics to supplied embeddings."""
    emb = np.array(emb)
    metric_values = dict()

    same, diff, average = same_diff_average(emb, labels)

    metric_values["ds"] = f"{(same / average):.4f}"
    metric_values["dd"] = f"{(diff / average):.4f}"

    to_centroid = dist_to_centroid(
        emb,
        labels,
        ds.class_to_idx.values()
    ) / average
    metric_values["cs"] = f"{to_centroid:.4f}"

    between_centroids = dist_between_centroids(
        emb, 
        labels, 
        ds.class_to_idx.values()
    ) / average 

    metric_values["cd"] = f"{between_centroids:.4f}"
    metric_values["acc_knn"] = f"{knn_evaluate(emb, labels):.4f}"
    
    return metric_values

def get_activations_labels(n_examples, width, height, train_dataloader, model, device):
    """Get activations of a neural network on supplied dataset"""
    activations_zero = np.zeros((n_examples, width * height))
    activations_first = [np.zeros((n_examples, 12*12)) for x in range(4)]
    activations_second = [np.zeros((n_examples, 5*5)) for x in range(16)]
    activations_third = np.zeros((n_examples, 64))

    labels = np.zeros((n_examples))

    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    index = 0
    with torch.no_grad():
        for X,y in train_dataloader:
            # Use GPU if possible
            X,y = X.to(device), y.to(device)

            batch_size = X.shape[0]
            acts = model.activations(X)

            # Activations in individual layers
            a_zero = acts[0]
            a_first = acts[1]
            a_second = acts[2]
            a_third = acts[3]

            # Correctly reshaping to batch_size x flattened rest, 
            # rewriting the zeros to the proper activations

            a_zero_reshaped = a_zero.reshape((batch_size, -1))
            activations_zero[index:(index+batch_size)] = a_zero_reshaped

            for i in range(4):
                a_first_reshaped = (a_first[:,i]).reshape((batch_size, -1))
                activations_first[i][index:(index+batch_size)] = a_first_reshaped

            for i in range(16):
                a_second_reshaped = (a_second[:,i]).reshape((batch_size, -1))
                activations_second[i][index:(index+batch_size)] = a_second_reshaped

            a_third_reshaped = a_third.reshape((batch_size, -1))
            activations_third[index:(index+batch_size)] = a_third_reshaped

            labels[index:(index+batch_size)] = y.detach().to("cpu").numpy()
            index += batch_size

    labels = labels.astype(np.int32)
    return activations_zero, activations_first, activations_second, activations_third, labels

def analyze_outliers(training_data, embeddings, labels, out_dir):
    """Create and save visualizations of outliers within the embedding"""
    for l1 in training_data.classes:
        for l2 in training_data.classes:
            if l1 == l2:
                continue

            # Get the label used in the dataset
            first = training_data.class_to_idx[l1]
            second = training_data.class_to_idx[l2]

            # Mask out other labels
            mask_first = training_data.targets == first
            mask_second = training_data.targets == second

            data_first = training_data.data[mask_first] 
            data_second = training_data.data[mask_second] 

            # Train KNN
            model = KNeighborsClassifier(n_neighbors=10)
            model.fit(embeddings, labels)
            pred = model.predict(embeddings)

            # Examples classified as `second` even though they are `first`
            indices_first_true = np.nonzero(labels == first)[0]
            indices_second_pred = np.nonzero(pred == second)[0]
            indices_first_second = np.intersect1d(indices_first_true, indices_second_pred) 
            outliers = training_data.data[indices_first_second]

            # Examples classified as `first` correctly
            indices_first_pred = np.nonzero(pred == first)[0]
            indices_first_correct = np.intersect1d(indices_first_true, indices_first_pred) 
            correct_first = training_data.data[indices_first_correct]

            # Examples classified as `second` correctly
            indices_second_true = np.nonzero(labels == second)[0]
            indices_second_pred = np.nonzero(pred == second)[0]
            indices_second_correct = np.intersect1d(indices_second_true, indices_second_pred) 
            correct_second = training_data.data[indices_second_correct]

            # Save plots
            if (len(outliers) > 0) and (len(correct_first) > 0) and (len(correct_second) > 0):
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,8))
                axes[0].imshow(correct_first[0])
                axes[1].imshow(correct_second[0])
                axes[2].imshow(outliers[0])

                plt.savefig(out_dir + "correct" + str(first) + "_predicted" + str(second) + ".png")
                plt.close()

def analyze_filters(training_data, out_dir, model, rng, device):
    """Create and save visualizations of activations in individual filters"""
    for l in training_data.classes:
        # Select random example
        val = training_data.class_to_idx[l]
        mask_val = training_data.targets == val
        data_val = training_data.data[mask_val]

        example = rng.choice(data_val, size=1)
        example = example[None,:]

        # Extract activations
        acts = model.activations(torch.Tensor(example).to(device))

        a_zero = acts[0].squeeze()
        a_first = acts[1].squeeze()
        a_second = acts[2].squeeze()
        a_third = acts[3].squeeze()

        # Original
        fig, axes = plt.subplots(figsize=(8,8))
        axes.imshow(a_zero)

        plt.savefig(out_dir + "class" + str(val) + "_l0.png")
        plt.close()

        # First layer
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5,20))
        for i,a in enumerate(a_first):
            axes[i].imshow(a)

        plt.savefig(out_dir + "class" + str(val) + "_l1.png")
        plt.close()

        # second layer
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12,12))
        for i in range(4):
            for j in range(4):
                idx = 4*i + j
                axes[i,j].imshow(a_second[idx])

        plt.savefig(out_dir + "class" + str(val) + "_l2.png")
        plt.close()

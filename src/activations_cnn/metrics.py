import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def same_diff_average(embeddings, labels):
    """
    Compute various average distances between points of embedding.

    Returns:
        same_class_dist (float): Average distance between points belonging to
                                 the same class
        diff_class_dist (float): Average distance between points belonging to
                                 different classes
        all_dist (float): Average distance between two points
    """
    # Total distance between points of the same class
    same_class_dist = 0
    
    # Total distance between points of different classes
    diff_class_dist = 0
    
    same_class_count = 0
    diff_class_count = 0
    
    # Compute for each pair of points
    for i in range(len(embeddings)-1):
        current = embeddings[i]
        remaining = embeddings[(i+1):]
        
        # Distance from single point to all the following points
        distances = np.sqrt(np.sum((current - remaining)**2, axis=1))
        
        # Separate distances for same class and for different classes
        same_class_indices = np.argwhere(labels[(i+1):] == labels[i])
        same_class_indices = np.squeeze(same_class_indices)
        diff_class_indices = np.argwhere(labels[(i+1):] != labels[i])
        diff_class_indices = np.squeeze(diff_class_indices)
        
        # Add distances to the total distance measured
        same_class_dist += np.sum(distances[same_class_indices])
        diff_class_dist += np.sum(distances[diff_class_indices])
            
        same_class_count += same_class_indices.size
        diff_class_count += diff_class_indices.size
        
    # By summing over distances for same and different classes, we get an average distance between
    # any two points in the embedding
    all_dist = (same_class_dist + diff_class_dist) / (same_class_count + diff_class_count)
    
    # Compute average distance between two points 
    same_class_dist /= same_class_count
    diff_class_dist /= diff_class_count
        
    return same_class_dist, diff_class_dist, all_dist

def dist_to_centroid(embeddings, labels, classes):
    """Compute average distance of a point to its class centroid."""

    total_distance = 0
    for c in classes:
        indices = np.argwhere(labels == c)
        indices = np.squeeze(indices)
        
        class_embeddings = embeddings[indices]
        centroid = np.sum(class_embeddings, axis=0) / len(indices)
        
        distances = np.sqrt(np.sum((class_embeddings - centroid)**2, axis=1))
        total_distance += np.sum(distances) / len(indices)
        
    return total_distance / len(classes)

def dist_between_centroids(embeddings, labels, classes):
    """Compute average distance between centroids of different classes."""

    # Compute centroids
    centroids = np.zeros((len(classes),2))
    for i,c in enumerate(classes):
        indices = np.argwhere(labels == c)
        indices = np.squeeze(indices)
        
        class_embeddings = embeddings[indices]
        centroids[i] = np.sum(class_embeddings, axis=0) / len(indices)
        
    # Calculate distances
    dist = 0
    count = 0
    
    for i in range(len(centroids)-1):
        current = centroids[i]
        remaining = centroids[(i+1):]
        
        distances = np.sqrt(np.sum((current - remaining)**2, axis=1))
        dist += np.sum(distances)
        count += len(remaining)
        
    return dist / count

def knn_evaluate(data, labels, K=10):
    """Compute accuracy of k-nn classifier on supplied data."""

    model = KNeighborsClassifier(n_neighbors = K)
    model.fit(data, labels)
    pred = model.predict(data)
    
    # Accuracy
    return np.sum(labels == pred) / len(data)

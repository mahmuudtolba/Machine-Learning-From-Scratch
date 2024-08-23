import numpy as np
import random
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        for i in range(self.max_iters):
            # Assign clusters based on closest centroid
            closest_centroids  = self._closest_centroid(X)
            
            # Plot the steps if enabled
            if self.plot_steps:
                self._plot(X, closest_centroids, i)
            
            
            # Update clusters and centroids
            clusters = self._update_clusters(X, closest_centroids)
            new_centroids = self._update_centroids( clusters)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f'Convergence reached after {i} iterations.')
                break

            self.centroids = new_centroids
            

        return closest_centroids
        
        
        
    def _initialize_centroids(self, X):
        indices = random.sample(range(X.shape[0]), self.K)
        return X[indices, :]

        

    def _closest_centroid(self, X):
        centroid_distances = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            centroid_distance = np.linalg.norm(X - self.centroids[k], axis=1)
            centroid_distances[:, k] = centroid_distance

        return np.argmin(centroid_distances, axis=1)
    

    def _update_clusters(self , X ,closest_centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, label in enumerate(closest_centroids):
            clusters[label].append(X[idx])

        return clusters


    def _update_centroids(self , clusters):
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.K):
            if clusters[k]:  # Check if the cluster is not empty
                new_centroids[k] = np.mean(clusters[k], axis=0)
        return new_centroids





    def _plot(self, X, labels, iteration):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=200, alpha=0.5, marker='X')
        plt.title(f'Iteration {iteration + 1}')
        plt.show()

# Testing
if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )

    clusters = len(np.unique(y))

    k = Kmeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.fit(X)

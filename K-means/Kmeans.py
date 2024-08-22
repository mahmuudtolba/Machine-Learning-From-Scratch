import numpy as np
import random
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # List of samples indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

    def fit(self, X):
        # Initialize the indices
        self.indices = random.sample(range(X.shape[0]), self.K)
        self.centroids = X[self.indices, :]
        
        for i in range(self.max_iters):
            # Get distances and assign clusters
            centroid_distances = self._get_distance(X)
            closest_centroids = self._closest_centroid(centroid_distances)
            
            # Check for convergence (optional)
            if hasattr(self, 'prev_centroids') and np.allclose(self.centroids, self.prev_centroids):
                print(f'Convergence reached after {i} iterations.')
                break
            
            # Save current centroids
            self.prev_centroids = self.centroids.copy()

            # Update clusters
            
            self.clusters = self._update_clusters(closest_centroids)

            # Update centroids
            self.centroids = self._update_centroids()
            
            if self.plot_steps:
                self._plot(X, closest_centroids, i)

        return closest_centroids

    def _get_distance(self, X):
        # Get the distance
        centroid_distances = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            centroid_distance = np.linalg.norm(X - self.centroids[k], axis=1)
            centroid_distances[:, k] = centroid_distance

        return centroid_distances

    def _closest_centroid(self, centroid_distance):
        return np.argmin(centroid_distance, axis=1)
    

    def _update_clusters(self , closest_centroids):
        self.clusters = [[] for _ in range(self.K)]
        for idx, label in enumerate(closest_centroids):
            self.clusters[label].append(X[idx])

        return self.clusters


    def _update_centroids(self):
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.K):
            if len(self.clusters[k]) > 0:
                new_centroids[k] = np.mean(self.clusters[k], axis=0)


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

import numpy as np  

# step 1 : randomly initialize the centroids
def initialize_centroids(X , K):
	np.random.seed(42)
	indices = np.random.choice(X.shape[0] , K , replace = True)
	return X[indices]

# step 2 : 
def assign_clusters(X , centroids):
	distances = np.linalg.norm(X[: , np.newaxis] - centroids , axis = 2)
	return np.argmin(distances , axis = 1)


def update_centroids(X , labels , K):
	centroids = np.array([X[labels == i].mean(axis = 0) for i in range(K)])
	return centroids

def compute_inertia(X, labels ,  centroids):
	""" Compute the mse from samples to it's cloest centroids """
	distances = np.linalg.norm(X  - centroids[labels] , axis = 1)
	return np.sum(distances ** 2)


def Kmeans(X , K , max_iterations , tol= 1e-4):
	centroids = initialize_centroids(X , K)

	for _ in range(max_iterations):
		
		labels = assign_clusters(X , centroids)

		new_centroids = update_centroids(X , labels , K)


		# step 4: check for convergence

		if np.all(np.abs(new_centroids - centroids) < tol):
			break



		centroids = new_centroids


	return centroids , labels , compute_inertia(X , labels , centroids)




# Example usage with dummy data
np.random.seed(42)
data = np.random.rand(100, 3)  # 100 samples, 2 features

k = 3  # Number of clusters
centroids, labels , mse = kmeans(data, k)

print("Final centroids:\n", centroids)
print("Cluster labels:\n", labels)
print("model inertia_:\n", mse)

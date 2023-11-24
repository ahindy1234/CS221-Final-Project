import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

def adaptive_kmeans_compression(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    X_compressed = kmeans.transform(X)
    X_reconstructed = np.zeros_like(X)
    for i in range(n_clusters):
        X_reconstructed[kmeans.labels_ == i] = kmeans.cluster_centers_[i]
    return X_compressed, X_reconstructed

def evaluate_compression(X, X_reconstructed):
    mse = mean_squared_error(X, X_reconstructed)
    compression_ratio = X.size / X_reconstructed.size
    return mse, compression_ratio

# Load CIFAR-10
cifar10 = load_digits()  # Change this to load CIFAR-10

# Preprocess the data (you may need to reshape or preprocess differently for CIFAR-10)
X = cifar10.data
y = cifar10.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define values of k (n_clusters) to test
k_values = [4, 8, 16, 32, 64]

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    mse_values = []
    compression_ratio_values = []
    
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]

        # Apply adaptive k-means compression
        X_train_compressed, X_train_reconstructed = adaptive_kmeans_compression(X_train, k)
        X_test_compressed, X_test_reconstructed = adaptive_kmeans_compression(X_test, k)

        # Evaluate compression
        mse, compression_ratio = evaluate_compression(X_test, X_test_reconstructed)
        mse_values.append(mse)
        compression_ratio_values.append(compression_ratio)

    # Print average results for the current k
    avg_mse = np.mean(mse_values)
    avg_compression_ratio = np.mean(compression_ratio_values)
    print(f"Average MSE for k={k}: {avg_mse}, Average Compression Ratio: {avg_compression_ratio}")

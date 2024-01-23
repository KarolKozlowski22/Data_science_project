import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download Fashion-MNIST dataset from OpenML
fashion_mnist = fetch_openml(data_id=40996, parser='auto', cache=True, as_frame=False)

# Extract data and labels
X, y = fashion_mnist.data, fashion_mnist.target.astype(int)

# Define class names for reference
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Reduce the size of the dataset (e.g., using 50% of the data)
X_sampled, _, y_sampled, _ = train_test_split(X, y, test_size=0.7, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sampled)

# Summary of the data
num_samples, num_features = X_scaled.shape
num_classes = len(np.unique(y_sampled))

print(f"Number of samples: {num_samples}")
print(f"Number of features (pixels per image): {num_features}")
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Visualize class distribution
plt.figure(figsize=(10, 5))
plt.hist(y_sampled, bins=num_classes, rwidth=0.8)
plt.xticks(np.arange(num_classes), class_names, rotation=45)
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Class Distribution")
plt.show()

# Display sample images from each class
plt.figure(figsize=(12, 12))
for i in range(num_classes):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    class_indices = np.where(y_sampled == i)[0]
    sample_index = np.random.choice(class_indices)
    plt.imshow(X_sampled[sample_index].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(class_names[i])
plt.show()

# Reduce data dimensionality using PCA and SVD
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_scaled)

# Visualize the reduced data using PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sampled, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(label="Class")
plt.title("PCA")

# Visualize the reduced data using TruncatedSVD
plt.subplot(1, 2, 2)
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y_sampled, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(label="Class")
plt.title("TruncatedSVD")

#save these images to file 
plt.savefig('gpt_pca_svd.png')

# Apply K-Means clustering
n_clusters = len(np.unique(y_sampled))  # Number of clusters equals the number of classes
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

# Evaluate clustering results using Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y_sampled, cluster_labels)
print(f"Adjusted Rand Index (ARI): {ari}")

# Visualize the clustering results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sampled, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(label="True Class")
plt.title("True Classes")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap=plt.cm.get_cmap("jet", n_clusters))
plt.colorbar(label="Cluster")
plt.title("K-Means Clusters")

#save these images to file
plt.savefig('gpt_kmeans.png')

# Split the sampled dataset into training and testing sets (e.g., 80% for training and 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sampled, test_size=0.2, random_state=42)

# Train a logistic regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


with open('gpt_output.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}")
    f.write("\nConfusion Matrix:")
    f.write(str(conf_matrix))
    f.write("\nClassification Report:")
    f.write(str(class_report))

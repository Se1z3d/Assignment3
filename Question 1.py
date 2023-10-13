from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Fetch the Olivetti faces dataset
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)

# The dataset is now loaded into the 'olivetti_faces' variable
data = olivetti_faces.data  # The image data
target = olivetti_faces.target  # The corresponding labels (subject IDs)

# Define the number of splits
n_splits = 1  # You can adjust this as needed

# Initialize the StratifiedShuffleSplit
stratified_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

# Split the data into train, validation, and test sets
for train_index, test_index in stratified_splitter.split(data, target):
    train_data, test_data = data[train_index], data[test_index]
    train_target, test_target = target[train_index], target[test_index]

# Further split the training data into a validation set
stratified_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=42)

for train_index, val_index in stratified_splitter.split(train_data, train_target):
    train_data, val_data = train_data[train_index], train_data[val_index]
    train_target, val_target = train_target[train_index], train_target[val_index]

# Define the classifier (linear in this example)
classifier = SVC(kernel='linear', C=1)

# Define the number of folds for cross-validation
n_splits = 5  # You can adjust the number of folds as needed

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

# Lists to store the validation accuracy for each fold
validation_accuracies = []

for train_index, val_index in stratified_kfold.split(train_data, train_target):
    # Split the training data into training and validation subsets for each fold
    train_subset, val_subset = train_data[train_index], train_data[val_index]
    train_labels, val_labels = train_target[train_index], train_target[val_index]

    # Train the classifier on the training subset
    classifier.fit(train_subset, train_labels)

    # Make predictions on the validation subset
    val_predictions = classifier.predict(val_subset)

    # Calculate the accuracy for this fold
    accuracy = accuracy_score(val_labels, val_predictions)
    validation_accuracies.append(accuracy)

# Calculate the mean accuracy and standard deviation across folds
mean_accuracy = sum(validation_accuracies) / len(validation_accuracies)
std_deviation = (sum([(x - mean_accuracy) ** 2 for x in validation_accuracies]) / len(validation_accuracies)) ** 0.5

print("Mean Validation Accuracy:", mean_accuracy)
print("Standard Deviation:", std_deviation)

# Step 1: Dimensionality Reduction using PCA for Euclidean Distance
n_components = 2  # Number of components after PCA
pca_euclidean = PCA(n_components=n_components)
reduced_data_pca_euclidean = pca_euclidean.fit_transform(data)

# Step 2: Hierarchical Clustering on the PCA-reduced data for Euclidean Distance
n_clusters_euclidean = 3  # Number of clusters
agg_clustering_pca_euclidean = AgglomerativeClustering(n_clusters=n_clusters_euclidean, linkage='ward', affinity='euclidean')
cluster_labels_pca_euclidean = agg_clustering_pca_euclidean.fit_predict(reduced_data_pca_euclidean)

# Step 3: Visualization
plt.scatter(reduced_data_pca_euclidean[:, 0], reduced_data_pca_euclidean[:, 1], c=cluster_labels_pca_euclidean, cmap='rainbow')
plt.title('Hierarchical Clustering with PCA for Euclidean Distance')
plt.show()

# Step 4: Evaluate the clustering using silhouette score for Euclidean Distance
silhouette_avg_pca_euclidean = silhouette_score(reduced_data_pca_euclidean, cluster_labels_pca_euclidean)
print(f'Silhouette Score (PCA for Euclidean Distance): {silhouette_avg_pca_euclidean}')

# Step 5: Dimensionality Reduction using t-SNE for Euclidean Distance
n_components_tsne_euclidean = 2  # Number of components after t-SNE
tsne_euclidean = TSNE(n_components=n_components_tsne_euclidean, metric="euclidean")
reduced_data_tsne_euclidean = tsne_euclidean.fit_transform(data)

# Step 6: Hierarchical Clustering on the t-SNE-reduced data for Euclidean Distance
n_clusters_tsne_euclidean = 3  # Number of clusters
agg_clustering_tsne_euclidean = AgglomerativeClustering(n_clusters=n_clusters_tsne_euclidean, linkage='ward', affinity='euclidean')
cluster_labels_tsne_euclidean = agg_clustering_tsne_euclidean.fit_predict(reduced_data_tsne_euclidean)

# Step 7: Visualization
plt.scatter(reduced_data_tsne_euclidean[:, 0], reduced_data_tsne_euclidean[:, 1], c=cluster_labels_tsne_euclidean, cmap='rainbow')
plt.title('Hierarchical Clustering with t-SNE for Euclidean Distance')
plt.show()

# Step 8: Evaluate the clustering using silhouette score for Euclidean Distance
silhouette_avg_tsne_euclidean = silhouette_score(reduced_data_tsne_euclidean, cluster_labels_tsne_euclidean)
print(f'Silhouette Score (t-SNE for Euclidean Distance): {silhouette_avg_tsne_euclidean}')

# Step 9: Dimensionality Reduction using PCA for Minkowski Distance
n_components_pca_minkowski = 2  # Number of components after PCA
pca_minkowski = PCA(n_components=n_components_pca_minkowski)
reduced_data_pca_minkowski = pca_minkowski.fit_transform(data)

# Step 10: Hierarchical Clustering on the PCA-reduced data for Minkowski Distance
n_clusters_pca_minkowski = 3  # Number of clusters
agg_clustering_pca_minkowski = AgglomerativeClustering(n_clusters=n_clusters_pca_minkowski, linkage='average', affinity='manhattan')
cluster_labels_pca_minkowski = agg_clustering_pca_minkowski.fit_predict(reduced_data_pca_minkowski)

# Step 11: Visualization
plt.scatter(reduced_data_pca_minkowski[:, 0], reduced_data_pca_minkowski[:, 1], c=cluster_labels_pca_minkowski, cmap='rainbow')
plt.title('Hierarchical Clustering with PCA for Minkowski Distance')
plt.show()

# Step 12: Evaluate the clustering using silhouette score for Minkowski Distance
silhouette_avg_pca_minkowski = silhouette_score(reduced_data_pca_minkowski, cluster_labels_pca_minkowski)
print(f'Silhouette Score (PCA for Minkowski Distance): {silhouette_avg_pca_minkowski}')

# Step 13: Dimensionality Reduction using t-SNE for Minkowski Distance
n_components_tsne_minkowski = 2  # Number of components after t-SNE
tsne_minkowski = TSNE(n_components=n_components_tsne_minkowski, metric="manhattan")
reduced_data_tsne_minkowski = tsne_minkowski.fit_transform(data)

# Step 14: Hierarchical Clustering on the t-SNE-reduced data for Minkowski Distance
n_clusters_tsne_minkowski = 3  # Number of clusters
agg_clustering_tsne_minkowski = AgglomerativeClustering(n_clusters=n_clusters_tsne_minkowski, linkage='average', affinity='manhattan')
cluster_labels_tsne_minkowski = agg_clustering_tsne_minkowski.fit_predict(reduced_data_tsne_minkowski)

# Step 15: Visualization
plt.scatter(reduced_data_tsne_minkowski[:, 0], reduced_data_tsne_minkowski[:, 1], c=cluster_labels_tsne_minkowski, cmap='rainbow')
plt.title('Hierarchical Clustering with t-SNE for Minkowski Distance')
plt.show()

# Step 16: Evaluate the clustering using silhouette score for Minkowski Distance
silhouette_avg_tsne_minkowski = silhouette_score(reduced_data_tsne_minkowski, cluster_labels_tsne_minkowski)
print(f'Silhouette Score (t-SNE for Minkowski Distance): {silhouette_avg_tsne_minkowski}')

# Step 17: Dimensionality Reduction using PCA for Cosine Similarity
n_components_pca_cosine = 2  # Number of components after PCA
pca_cosine = PCA(n_components=n_components_pca_cosine)
reduced_data_pca_cosine = pca_cosine.fit_transform(data)

# Step 18: Hierarchical Clustering on the PCA-reduced data for Cosine Similarity
n_clusters_pca_cosine = 3  # Number of clusters
agg_clustering_pca_cosine = AgglomerativeClustering(n_clusters=n_clusters_pca_cosine, linkage='average', affinity='cosine')
cluster_labels_pca_cosine = agg_clustering_pca_cosine.fit_predict(reduced_data_pca_cosine)

# Step 19: Visualization
plt.scatter(reduced_data_pca_cosine[:, 0], reduced_data_pca_cosine[:, 1], c=cluster_labels_pca_cosine, cmap='rainbow')
plt.title('Hierarchical Clustering with PCA for Cosine Similarity')
plt.show()

# Step 20: Evaluate the clustering using silhouette score for Cosine Similarity
silhouette_avg_pca_cosine = silhouette_score(reduced_data_pca_cosine, cluster_labels_pca_cosine)
print(f'Silhouette Score (PCA for Cosine Similarity): {silhouette_avg_pca_cosine}')

# Step 21: Dimensionality Reduction using t-SNE for Cosine Similarity
n_components_tsne_cosine = 2  # Number of components after t-SNE
tsne_cosine = TSNE(n_components=n_components_tsne_cosine, metric="cosine")
reduced_data_tsne_cosine = tsne_cosine.fit_transform(data)

# Step 22: Hierarchical Clustering on the t-SNE-reduced data for Cosine Similarity
n_clusters_tsne_cosine = 3  # Number of clusters
agg_clustering_tsne_cosine = AgglomerativeClustering(n_clusters=n_clusters_tsne_cosine, linkage='average', affinity='cosine')
cluster_labels_tsne_cosine = agg_clustering_tsne_cosine.fit_predict(reduced_data_tsne_cosine)

# Step 23: Visualization
plt.scatter(reduced_data_tsne_cosine[:, 0], reduced_data_tsne_cosine[:, 1], c=cluster_labels_tsne_cosine, cmap='rainbow')
plt.title('Hierarchical Clustering with t-SNE for Cosine Similarity')
plt.show()

# Step 24: Evaluate the clustering using silhouette score for Cosine Similarity
silhouette_avg_tsne_cosine = silhouette_score(reduced_data_tsne_cosine, cluster_labels_tsne_cosine)
print(f'Silhouette Score (t-SNE for Cosine Similarity): {silhouette_avg_tsne_cosine}')

#!/usr/bin/env python3
from project_steps.data_split import get_and_split_data
from project_steps.data_summary import data_summary
from project_steps.PCA_reduction import pca_2d_reduction, pca_3d_reduction
from project_steps.visualize import visualize_2d, visualize_3d
from project_steps.cluster import cluster_kmeans_accuracy
from project_steps.truncatedSVD import truncatedSVD_2d_reduction, truncatedSVD_3d_reduction
from project_steps.tSNE import tsne_2d_reduction, tsne_3d_reduction
from project_steps.classification import classify_evalute

(train_images, train_labels), (test_images, test_labels) = get_and_split_data()
(summary_images, summary_labels), (test_summary_images, test_summary_labels) = get_and_split_data(sample_size=0.001)
count_num_of_each_label = [0] * 10

for label in train_labels:
    count_num_of_each_label[label] += 1

print(count_num_of_each_label)

train_images_flat = train_images.reshape((train_images.shape[0], -1))
test_images_flat = test_images.reshape((test_images.shape[0], -1))
summary_images_flat = summary_images.reshape((summary_images.shape[0], -1))
data_summary(summary_images_flat, summary_labels)

train_images_pca_2d = pca_2d_reduction(train_images_flat)
train_images_pca_3d = pca_3d_reduction(train_images_flat)

train_images_truncatedSVD_2d = truncatedSVD_2d_reduction(train_images_flat)
train_images_truncatedSVD_3d = truncatedSVD_3d_reduction(train_images_flat)

# train_images_tsne_2d = tsne_2d_reduction(train_images_flat)
# train_images_tsne_3d = tsne_3d_reduction(train_images_flat)

visualize_2d(train_images_pca_2d, train_labels, 'PCA')
visualize_3d(train_images_pca_3d, train_labels, 'PCA')

# visualize_2d(train_images_tsne_2d, train_labels, 't-SNE')
# visualize_3d(train_images_tsne_3d, train_labels, 't-SNE')

visualize_2d(train_images_truncatedSVD_2d, train_labels, 'Truncated SVD')
visualize_3d(train_images_truncatedSVD_3d, train_labels, 'Truncated SVD')

cluster_kmeans_accuracy(train_images_flat, train_labels)

classify_evalute(train_images_flat, train_labels, test_images_flat, test_labels)



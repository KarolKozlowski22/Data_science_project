from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

def cluster_kmeans_accuracy(train_images_flat, train_labels):
    kmeans = KMeans(n_clusters=10, n_init=10)
    clusters = kmeans.fit_predict(train_images_flat)
    labels = np.zeros_like(clusters)
    for i in range(10):
        mask = (clusters == i)
        labels[mask] = np.argmax(np.bincount(train_labels[mask]))

    conf_matrix = confusion_matrix(train_labels, labels)
    accuracy = accuracy_score(train_labels, labels)
    silhouette = silhouette_score(train_images_flat, clusters)
    ari = adjusted_rand_score(train_labels, clusters)
    homogeneity = homogeneity_score(train_labels, clusters)
    completeness = completeness_score(train_labels, clusters)
    v_measure = v_measure_score(train_labels, clusters)

    with open('output.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\nAccuracy: {:.2f}%".format(accuracy * 100))
        f.write("\nSilhouette Score: {:.2f}".format(silhouette))
        f.write("\nAdjusted Rand Index: {:.2f}".format(ari))
        f.write("\nHomogeneity Score: {:.2f}".format(homogeneity))
        f.write("\nCompleteness Score: {:.2f}".format(completeness))
        f.write("\nV-measure Score: {:.2f}".format(v_measure))


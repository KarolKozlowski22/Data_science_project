from tensorflow.keras.datasets import fashion_mnist
import numpy as np

def get_and_split_data(sample_size=0.1):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_sample_size = int(len(train_images) * sample_size)
    test_sample_size = int(len(test_images) * sample_size)

    train_indices = np.random.choice(range(len(train_images)), train_sample_size, replace=False)
    test_indices = np.random.choice(range(len(test_images)), test_sample_size, replace=False)

    train_images_sample = train_images[train_indices]
    train_labels_sample = train_labels[train_indices]
    test_images_sample = test_images[test_indices]
    test_labels_sample = test_labels[test_indices]

    return (train_images_sample, train_labels_sample), (test_images_sample, test_labels_sample)
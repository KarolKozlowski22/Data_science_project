from sklearn.decomposition import PCA


def pca_2d_reduction(train_images_flat):
    pca = PCA(n_components=2)
    train_images_pca = pca.fit_transform(train_images_flat)
    return train_images_pca

def pca_3d_reduction(train_images_flat):
    pca = PCA(n_components=3)
    train_images_pca = pca.fit_transform(train_images_flat)
    return train_images_pca


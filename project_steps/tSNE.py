from sklearn.manifold import TSNE

def tsne_2d_reduction(train_images_flat):
    tsne = TSNE(n_components=2)
    train_images_tsne = tsne.fit_transform(train_images_flat)
    return train_images_tsne

def tsne_3d_reduction(train_images_flat):
    tsne = TSNE(n_components=3)
    train_images_tsne = tsne.fit_transform(train_images_flat)
    return train_images_tsne
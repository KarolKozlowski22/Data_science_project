from sklearn.decomposition import TruncatedSVD

def truncatedSVD_2d_reduction(train_images_flat):
    truncatedSVD = TruncatedSVD(n_components=2)
    train_images_truncatedSVD = truncatedSVD.fit_transform(train_images_flat)
    return train_images_truncatedSVD

def truncatedSVD_3d_reduction(train_images_flat):
    truncatedSVD = TruncatedSVD(n_components=3)
    train_images_truncatedSVD = truncatedSVD.fit_transform(train_images_flat)
    return train_images_truncatedSVD

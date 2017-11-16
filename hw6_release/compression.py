import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    
    U, s, V = np.linalg.svd(image)
    print(U[:, :num_values].shape)
    print(np.diag(s[:num_values]).shape)
    print(V[:num_values, :].shape)
    
    
    
    compressed_image = np.matrix(U[:, :num_values]) * np.matrix(np.diag(s[:num_values])) * np.matrix(V[:num_values, :])
    
    H, W = compressed_image.shape
    
    compressed_size = num_values*H + num_values + num_values*W

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size

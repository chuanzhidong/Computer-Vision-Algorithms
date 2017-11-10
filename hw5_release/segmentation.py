import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage import color
from skimage import io
from skimage.util import img_as_float
from scipy.stats import entropy

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False) #array of k values with the values in range of N
    centers = features[idxs] #specific feature vectors chosen as centers
    assignments = np.zeros(N)
    previous_centers = centers*0.0
    
    for n in range(num_iters):
        
        if np.allclose(centers, previous_centers):
            break
        
        previous_centers = centers.copy()
            
        #Loop through every feature vector in features array
        for i in range(N):
            distances = cdist([features[i]], centers)
            min_index = np.argmin(distances)
            assignments[i] = min_index
        
        for i in range(k):
            indices = np.where(assignments == i)
            centers[i] = np.mean(features[indices], axis=0)

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    previous_centers = centers*0.0
    
    for n in range(num_iters):
        if np.allclose(centers, previous_centers,):
            break
        previous_centers = centers.copy()
        assignments = np.argmin(cdist(features, centers), axis = 1)
        for i in range(k):
            centers[i] = np.mean(features[np.where(assignments == i)], axis=0)
    return assignments


def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to defeine distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        dist = squareform(pdist(centers))
        dist[dist==0] = np.inf
        i, j = np.unravel_index(np.argmin(dist), dist.shape)

        assignments[np.where(assignments == j)] = i
        centers[assignments == i] = np.mean(features[assignments == i], axis = 0)
        n_clusters -= 1
        
    idxs = np.unique(assignments)
    for i in range(len(idxs)):
        assignments[assignments == idxs[i]] = i
        
        
        
    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    features = np.reshape(img, (H*W, C))

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    features[:,0:C] = img.reshape(H*W, C)
    grid = np.mgrid[0:H,0:W]
    grid_values = np.zeros((H,W,2))
    grid_values[:,:,0] = grid[0]
    grid_values[:,:,1] = grid[1]
    features[:,C:C+2] = grid_values.reshape(H*W, 2)
    
    features = (features - np.mean(features,axis=0))/np.std(features, axis=0)

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    
    H, W, C = img.shape
    features = np.zeros((H*W, C+2))
    
    features[:,0:C] = img.reshape(H*W, C)
    
    img_gray = color.rgb2gray(img)
    
    G, theta = gradient(img_gray)
    entro = np.zeros((H,W))
    padded = np.pad(theta, ((15,15),(15,15)), mode = 'constant')
    for i in range(H):
        for j in range(W):
            temp = np.histogram(padded[i:i+15, j:j+15], 8)[0]
            entro[i][j] = entropy(temp)
    features[:, C] = entro.reshape((H*W))
    
    kernel = np.zeros((20, 20))
    kernel.fill(1)
    weighted_gradient = conv(img_gray, kernel)
    features[:, C+1] = weighted_gradient.reshape((H*W))
    
    features = (features - np.mean(features,axis=0))/np.std(features, axis=0)


    return features
    
def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    
    G = np.sqrt(partial_x(img)**2 + partial_y(img)**2)
    theta = (np.arctan2(partial_y(img), partial_x(img))*180/np.pi + 360)%360
    

    return G, theta

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    for m in range(Hi):
        for n in range(Wi):
            image_splice = padded[m:m+Hk, n:n+Wk]
            #kernel_t = np.transpose(kernel)
            out[m][n] = np.sum(image_splice * np.flipud(np.fliplr(kernel)))
    return out

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    kernel_x = np.array(
    [[ 0, 0, 0],
     [ 0.5, 0, -0.5],
     [ 0, 0, 0]])
    out = conv(img, kernel_x)

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    kernel_y = np.array(
    [[ 0, 0.5, 0],
     [ 0, 0, 0],
     [ 0, -0.5, 0]])
    out = conv(img, kernel_y)

    return out

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    
    H,W = mask_gt.shape
    
    TP = (mask_gt*mask).sum()
    TN = ((1-mask_gt)*(1-mask)).sum()
    
    accuracy = np.count_nonzero(mask_gt == mask)/(H*W)
    
    #accuracy = (TP+TN)/(P+N)

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy

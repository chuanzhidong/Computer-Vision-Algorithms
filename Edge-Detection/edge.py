import numpy as np

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
            out[m][n] = np.sum(image_splice * np.flipud(np.fliplr(kernel)))
    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))
    k = (size-1)//2

    for i in range(size):
        for j in range(size):
            kernel[i][j] = (1.0/(2*np.pi*sigma**2))*np.exp(-((i-k)**2 + (j-k)**2)/(2.0*sigma**2))
                        
    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

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


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
        
        theta[i][j]%180
        
        d_map = {0, [[1,0],[-1,0]]
        
        loop through every element in d_map
        
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    
    direction_map = {0: [[-1,0],[1,0]], 45: [[1,1],[-1,-1]], 90: [[0,1],[0,-1]], 135: [[-1,1],[1,-1]]}

    for i in range(H):
        for j in range(W):
            d = theta[i][j]
            supression_points = direction_map[d%180]
            out[i][j] = G[i][j]
            for point in supression_points:                
                if (i + point[1]) not in range(H) or (j + point[0]) not in range(W):
                    continue
                if G[i][j] <= G[i+point[1]][j+point[0]]:
                    out[i][j] = 0

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)
    H,W = img.shape
    
    for i in range(H):
        for j in range(W):
            strong_edges[i][j] = False
            weak_edges[i][j] = False
            if img[i][j] > high:
                strong_edges[i][j] = True
            elif img[i][j] < high and img[i][j] > low:
                weak_edges[i][j] = True
    
    

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """
    
    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    print(indices)
    edges = np.zeros((H, W))
    import queue
    
    q = queue.Queue()
    
    for point in indices:
        q.put(point)
    
    while not q.empty():
        s_edge = q.get()
        edges[s_edge[0]][s_edge[1]] = 1
        
        neighbor_list = get_neighbors(s_edge[0], s_edge[1], H, W)
        
        for temp in neighbor_list:
            i = temp[0]
            j = temp[1]
            
            if edges[i][j] == 1:
                continue
            
            if weak_edges[i][j] == 1:
                edges[i][j] = 1
                q.put(temp)

    return edges

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    h_other = Hk//2
    w_other = Wk//2
    
    conv_shape = (Hi + Hk - 1, Wi + Wk - 1)
    
   
    temp_1 = np.fft.rfft2(image, conv_shape)
    temp_2 = np.fft.rfft2(kernel, conv_shape)
    temp_3 = np.fft.irfft2(temp_1*temp_2, conv_shape)
    
    out = temp_3.astype(float)[Hk//2:Hi+Hk//2, Wk//2:Wi+Wk//2]

    return out

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    gaus_kernel = gaussian_kernel(kernel_size, sigma)
    out = conv(img, gaus_kernel)
    G, theta = gradient(out)
    out = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(out, high, low)
    edge = link_edges(strong_edges, weak_edges)

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    
    for x,y in zip(xs, ys):
        for i in range(num_thetas):
            rho = x*cos_t[i] + y*sin_t[i]
            accumulator[int(rho+diag_len), i] += 1
    

    return accumulator, rhos, thetas

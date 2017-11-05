import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for m in range(0, Hi):
        for n in range(0, Wi):
            for i in range(0, Hk):
                for j in range(0, Wk):
                    
                    x,y = m+Hk//2-i, n+Wk//2-j
                    
                    if (m+Hk//2-i < 0 or n+Wk//2-j < 0 or m+Hk//2-i >= Hi or n+Wk//2-j >= Wi):
                        continue
                    out[m][n] += image[m+Hk//2-i][n+Wk//2-j] * kernel[i][j]
                    

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None
    out = np.pad(image, ((pad_height, pad_height),(pad_width, pad_width)), "constant")
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    new_image = zero_pad(image, Hk//2, Wk//2)
    
    for m in range(Hi):
        for n in range(Wi):
            image_splice = new_image[m:m+Hk, n:n+Wk]
            out[m][n] = np.sum(image_splice * np.flipud(np.fliplr(kernel)))

    return out

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

    h_other = Hk//2
    w_other = Wk//2
    
    conv_shape = (Hi + Hk - 1, Wi + Wk - 1)
    
   
    temp_1 = np.fft.rfft2(image, conv_shape)
    temp_2 = np.fft.rfft2(kernel, conv_shape)
    temp_3 = np.fft.irfft2(temp_1*temp_2, conv_shape)
    
    out = temp_3.astype(float)[Hk//2:Hi+Hk//2, Wk//2:Wi+Wk//2]
    
    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    
    g_flip = np.flipud(np.fliplr(g))
    out = conv_fast(f, g_flip)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    g_mean = g-np.mean(g)
    out = cross_correlation(f, g_mean)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    new_image = zero_pad(f, Hk//2, Wk//2)
    
    for m in range(Hi):
        for n in range(Wi):
            image_splice = new_image[m:m+Hk, n:n+Wk]
            
            normalized_splice = (image_splice - np.mean(image_splice))/np.std(image_splice)
            normalized_g = (g-np.mean(g))/np.std(g)
            
            
            out[m][n] = np.sum(normalized_splice * normalized_g)

    return out

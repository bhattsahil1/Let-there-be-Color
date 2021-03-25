import numpy as np
from skimage import color

def net2RGB(L,ab):
    """Converts net output back to an image.
    That is, input is in the form of normalized L, ab channels. The function first 
    unnormalizes them, stacks them back to LAB space and it is finally converted to RGB space

    Args:
        L: original L channel of input image
        ab: ab channel learnt from our training network

    Returns:
        3 channel RGB image
    """

    L = 100 * L.numpy()             # Converting to numpy and unnormalize
    L = np.transpose(L,(1,2,0))     # Tranposing axis from Height * Width * Channels to H * W * C
    
    ab = 254 * ab.numpy() - 127     # Converting to numpy and unnormalize
    ab = np.transpose(ab,(1,2,0))   # Tranposing axis from Height * Width * Channels to H * W * C

    img = np.dstack(L,ab).astype('float64')
    return color.lab2rgb(img)
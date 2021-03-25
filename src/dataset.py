import os
import torch
import numpy as np
import torchvision.datasets
import torchvision.transforms
from skimage import color, io
from random import randint


class HandleGrayscale(object):  ## Converts any grayscale input to rgb
    def __call__(self, image):
        if len(image.shape) < 3:
            image = color.gray2rgb(image)
        return image


class RandomCrop(object):   ##  Randomly crops an image to size x size
    def __init__(self, size=224):
        self.size = size
        
    def __call__(self, image):

        h, w, _ = image.shape
        assert min(h, w) >= self.size

        off_h = randint(0, h - self.size)
        off_w = randint(0, w - self.size)

        cropped = image[off_h:off_h+self.size, off_w:off_w+self.size]

        assert cropped.shape == (self.size, self.size, 3)
        return cropped

    
class Rgb2LabNorm(object):
    def __call__(self, image):
        assert image.shape == (224, 224, 3)
        img_lab = color.rgb2lab(image)
        img_lab[:,:,:1] = img_lab[:,:,:1] / 100.0
        img_lab[:,:,1:] = (img_lab[:,:,1:] + 128.0) / 256.0
        return img_lab
    
    
class ToTensor(object):  ##     """Converts an image to torch.Tensor, image -> H*W*C to C*H*W
    def __call__(self, image):
        
        assert image.shape == (224, 224, 3)
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype(np.float32))
        assert image_tensor.shape == (3, 224, 224)
        return image_tensor

    
class SplitLab(object): ## Splits tensor LAB image to L and ab channels.
    def __call__(self, image):
        assert image.shape == (3, 224, 224)
        L  = image[:1,:,:]
        ab = image[1:,:,:]
        return (L, ab)
    
    
    
class ImagesDateset(torchvision.datasets.ImageFolder):

    def __init__(self, root, testing=False):
        super().__init__(root=root, loader=io.imread)
        
        self.testing = testing

        self.composed = torchvision.transforms.Compose(
            [HandleGrayscale(), RandomCrop(224), Rgb2LabNorm(), 
             ToTensor(), SplitLab()]
        )
        
            
        
    def __getitem__(self, idx):
        image, label =  super().__getitem__(idx)
        
        L, ab = self.composed(image)

        if self.testing:
            path = os.path.normpath(self.imgs[idx][0])
            name = os.path.basename(path)
            l1 = os.path.basename(os.path.dirname(path))
            label = l1 + "-" + name

        return L, ab, label
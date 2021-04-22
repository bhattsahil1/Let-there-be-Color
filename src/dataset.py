import cv2, os
import torch, torchvision.datasets, torchvision.transforms
import numpy as np
from skimage import color, io
from random import randint



class getGrayscale(object):  ## Converts any grayscale input to rgb
    def __call__(self, img):
        if len(img.shape) < 3:
            img = color.gray2rgb(img)
        return img


class GetLabNorm(object):   ##  Resize the image and convert the image to Lab Norm 
    def __init__(self, size=224):
        self.size = size        
    def __call__(self, img):
        h, w, c = img.shape
        t = img
        assert min(h, w) >= self.size
        resized = cv2.resize(img, (self.size, self.size))
        assert resized.shape == (self.size, self.size, 3)
        img_lab = color.rgb2lab(resized)
        img_lab[:,:,:1], img_lab[:,:,1:] = img_lab[:,:,:1] / 100.0, (img_lab[:,:,1:] + 128.0) / 256.0
        return img_lab
    
class SplitTheLab(object):     ##  Converts an image to torch.Tensor, image -> H*W*C to C*H*W and split the image to L and ab
    def __call__(self, img):    
        assert img.shape == (224, 224, 3)
        tensor_img = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype(np.float32))
        assert tensor_img.shape == (3, 224, 224)
        L, ab  = tensor_img[:1,:,:], tensor_img[1:,:,:]
        return (L, ab)
    
class ImagesDateset(torchvision.datasets.ImageFolder):
    def __init__(self, root, testing=False):
        super().__init__(root=root, loader=io.imread)  
        self.composed = torchvision.transforms.Compose([getGrayscale(), 
                                    GetLabNorm(224), SplitTheLab()])  
        self.testing = testing

              
    def __getitem__(self, idx):
        img, label =  super().__getitem__(idx)
        L, ab = self.composed(img)

        if self.testing:
            path_name = self.imgs[idx][0]
            path = os.path.normpath(path_name)
            name = os.path.basename(path)
            temp = os.path.basename(os.path.dirname(path))
            label = temp + "-" + name

        return L, ab, label
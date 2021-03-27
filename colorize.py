from skimage import io, color
from src import colnet
from src import dataset
from src import utils

import os
import argparse
import torch
import torchvision.transforms
import numpy as np

def colorize(path,model):
    checkpt = torch.load(model,map_location=torch.device("cpu"))
    net_divisor = checkpt['net_divisor']
    classes = checkpt['classes']
    n_classes = len(classes)

    net = colnet.ColNet(num_classes=n_classes,net_divisor = net_divisor)
    net.load_state_dict(checkpt['model_state_dict'])

    transforms = torchvision.transforms.Compose(
        [dataset.HandleGrayscale(),
        dataset.RandomCrop(224),
        dataset.Rgb2LabNorm(), 
        dataset.ToTensor(), 
        dataset.SplitLab()])
    
    img = io.imread(path)

    L,ab = transforms(img)
    L_tensor = torch.from_numpy(np.expand_dims(L,axis=0))

    softmax = torch.nn.Softmax(dim=1)
    net.eval()
    with torch.no_grad():
        ab_out,pred = net(L_tensor)
        img_color = utils.net2RGB(L,ab_out[0])
        io.imsave("colorized-"+os.path.basename(path),img_color)

        sm = softmax(pred)
        probs = sm[0].numpy()
        probs_classes = sorted(zip(probs,classes),key=lamda x:x[0],reverse=True)

        print('Predicted labels: \n')
        for p,c in probs_classes[:10]:
            print("{:>7.2f}% \t{}".format(p*100.0, c))
        
parser = argparse.ArgumentParser(description="A script to colorize a photo")
parser.add_argument('image', help="Path to the image. RGB one will be converted to grayscale")
parser.add_argument('model', help="Path a *.pt model")
args = parser.parse_args()
print("[Warrning] Only 224x224 images are supported. Otherwise an image will be randomly cropped")
colorize(args.image, args.model)    
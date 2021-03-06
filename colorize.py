import os, sys, argparse, torch, torchvision.transforms
import numpy as np
import torch.nn as nn
from skimage import io, color
from src import colnet, dataset, utils

def getImageTransformed(img_path):
    composed_transforms = torchvision.transforms.Compose(
            [dataset.getGrayscale(), 
             dataset.GetLabNorm(224),
             dataset.SplitTheLab()]
        )  
    L, _ = composed_transforms(io.imread(img_path))
    L_tensor = torch.from_numpy(np.expand_dims(L, axis=0))
    return L, L_tensor

def main(img_path, model):
    if torch.cuda.is_available():
        device_name = torch.device("cuda")
    else:
        device_name = torch.device("cpu")
    ## load the checkpoint
    checkpoint = torch.load(model, map_location=device_name)
    classes = checkpoint['classes']
    ##print(checkpoint['model_state_dict'])
    ## get the model saved in colnet
    net = nn.DataParallel(colnet.ColNet(num_classes=len(classes)))
    net.load_state_dict(checkpoint['model_state_dict'])
    softmax = torch.nn.Softmax(dim=1)
    net.eval()
    L, L_tensor = getImageTransformed(img_path)
    with torch.no_grad():
        ab_out, predicted = net(L_tensor)
        img_colorized = utils.net2RGB(L, ab_out[0])
        ## saving the resultant image 
        io.imsave("result-" + os.path.basename(img_path), img_colorized)
        probs = softmax(predicted)[0].cpu().numpy()
        probs_and_classes = sorted(zip(probs, classes), key=lambda x: x[0], reverse=True)
        bestResults = probs_and_classes[:10]
        for p, c in bestResults:
            print("Class =>", c, "Percentage =>", str(p*100.0) + "%")
                 

if __name__ == "__main__":
    image_path = sys.argv[1]
    model = sys.argv[2] 
    main(image_path, model)

import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from skimage import io

from .colnet import ColNet
from .dataset import ImagesDateset
from .utils import net2RGB

class Train:
    def __init__(self,batch_size,epochs,img_dir_train, img_dir_val,img_dir_test,start_epoch=0,net_divisor=1,learning_rate=0.0001,
        model_checkpoint=None,models_dir='./model/',img_out_dir='./out',num_workers=4):

        if os.path.exists(models_dir) == False:
            os.makedirs(models_dir)
        if os.path.exists(img_out_dir) == False:
            os.makedirs(img_out_dir)
        if model_checkpoint:
            self.load_checkpoint(model_checkpoint)
        
        self.current_model_name = model_checkpoint
        self.best_val_loss = float("inf")
        self.best_model_dir = os.path.join(models_dir, 'colnet-the-best.pt')
        
        self.classes = self.trainloader.dataset.classes
        self.num_classes = len(self.classes)
        self.BATCH_SIZE = BATCH_SIZE
        self.net_divisor = net_divisor
        self.EPOCHS = epochs
        self.start_epoch = start_epoch

        self.img_dir_val = img_dir_val    
        self.devset = ImagesDateset(self.img_dir_val)
        self.devloader = DataLoader(self.devset, batch_size=self.BATCH_SIZE,
                                    shuffle=False, num_workers=num_workers)

        self.img_dir_train = self.img_dir_train
        self.trainset = ImagesDateset(self.img_dir_train)
        self.trainloader = DataLoader(self.trainset, batch_size=self.BATCH_SIZE, 
                                      shuffle=True, num_workers=num_workers)
        self.img_dir_test = self.img_dir_test
        self.testset = ImagesDateset(self.img_dir_test, testing=True)
        self.testloader = DataLoader(self.testset, batch_size=self.BATCH_SIZE,
                                     shuffle=False, num_workers=num_workers)
        
        self.loss_history = { "train": [], "val":[] }
        self.mse = nn.MSELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                                   else "cpu")        
        self.net = ColNet(net_divisor=net_divisor, num_classes=self.num_classes)
        self.net.to(self.device)
        print("Using {}\n".format(self.device))
        self.img_out_dir = img_out_dir
        self.models_dir = models_dir
    
    def loss(self,class_target,class_out,col_target,col_out):
        loss_class = self.ce(class_out,class_target)
        loss_col = self.mse(col_target,col_out)
        return loss_col + loss_class/300
    
    def train(self,epoch):
        epoch_loss = 0
        self.net.train()

        for idx in range(len(self.trainloader)):
            L,ab,label = self.trainloader[idx]
            L, ab, labels = L.to(self.device), ab.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            ab_out,labels_out = self.net(L)

            loss = self.loss(ab,ab_out,labels,labels_out)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            print('[Epoch {:>2} / {} | Batch: {:>2} / {}] loss: {:>10.3f}'.format(epoch+1, self.EPOCHS, idx + 1, len(self.trainloader), loss.item()))
        
        epoch_loss /= len(self.trainloader)
        self.loss_history['train'].append(epoch_loss)

        print("Epoch loss: ",epoch_loss)
    
    def validate(self,epoch):
        dev_loss = 0
        self.net.eval()
        with torch.no_grad():
            for idx in range(len(self.devloader)):
                L_dev,ab_dev,labels_dev = self.devloader[idx]
                L_dev,ab_dev,labels_dev = L_dev.to(self.device),ab_dev.to(self.device),labels_dev.to(self.device)
                
                ab_dev_out,labels_dev_out = self.net(L_dev)

                dev_batch_loss = self.loss(ab_dev,ab_dev_out,labels_dev,labels_dev_out)
                dev_loss += dev_batch_loss.item()

                print("[Validation] [Batch {:>2} / {}] dev loss: {:>10.3f}".format(idx+1, len(self.devloader), dev_batch_loss))
            
            dev_loss /= len(self.devloader)
            self.loss_history['val'].append(dev_loss)
            print("Dev loss ",dev_loss)
    
    def test(self,model_dir):
        if (model_dir is None):
            model_dir = self.current_model_name
            if os.path.isfile(self.best_model_dir):
                model_dir = self.best_model_dir
        
        self.load_checkpoint(model_dir)
        self.net.to(self.device)
        self.net.eval()
        
        with torch.no_grad():
            for idx in range(len(self.testloader)):
                L,_,labels = self.testloader[idx]
                L = L.to(self.device)
                ab_out,_ = self.net(L)

                L = L.to(torch.device("cpu"))
                ab_out = ab_out.to(torch.device("cpu"))

                for i in range(L.shape[0]):
                    img = net2RGB(L[i],ab_out[i])
                    io.imsave(os.path.join(self.img_out_dir,labels[i]),img)

        print("Saved all photos to "+self.img_out_dir)

    def save_checkpoint(self,epoch):
        full_path = os.path.join(self.models_dir,"colnet{}-{}.pt".format(time.strftime("%y%m%d-%H-%M-%S"), epoch))

        torch.save({
            'epoch':epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer_state_dict(),
            'losses': self.loss_history,
            'net_divisor': self.net_divisor,
            'classes': self.classes
        },full_path)

        self.current_model_name = full_path
        print('\nsaved model to {}\n'.format(full_path))

        current_val_loss = self.loss_history['val'][-1]
        if (current_val_loss < self.best_val_loss):
            self.best_val_loss = current_val_loss
            shutil.copy(full_path, self.best_model_dir)
            print("Saved the best model on epoch: ",epoch+1,"\n")

    def load_checkpoint(self,model_checkpoint):
        print('Resume training of: '+model_checkpoint)
        checkpoint = torch.load(model_checkpoint, map_location=torch.device("cpu"))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['losses']
        self.start_epoch = checkpoint['epoch'] + 1 
        self.net_divisor = checkpoint['net_divisor'] 
        self.current_model_name = model_checkpoint

    def run(self):
        for epoch in range(self.start_epoch,self.EPOCHS):
            print("{2}\nEpoch {0} / {1}\n{2}"
                  .format(epoch + 1, self.EPOCHS, '-'*47))
            self.train(epoch)
            self.validate(epoch)
            self.save_checkpoint(epoch)
        print('\nFinished Training.\n')
    
    def info(self):
        print("{0} Training environment info {0}\n".format("-"*13))

        print("Training starts from epoch: {}".format(self.start_epoch))
        print("Total number of epochs:     {}".format(self.EPOCHS))
        print("ColNet parameters are devided by: {}".format(self.net_divisor))
        print("Batch size:  {}".format(self.BATCH_SIZE))
        print("Used devide: {}".format(self.device))
        print("Number of classes: {}".format(self.num_classes))
        print()

        if self.current_model_name:
            print("Current model name:      " + self.current_model_name)

        print("Training data directory: " + self.img_dir_train)
        print("Validate data directory: " + self.img_dir_val)
        print("Testing data directory:  " + self.img_dir_test)
        print("Models are saved to:     " + self.models_dir)
        print("Colorized images are saved to: " + self.img_out_dir)
        print("-" * 53 + "\n")

if __name__ == "__main__":
    print("Hello, have a great day!")
            
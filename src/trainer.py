import os, shutil, time, torch
import torch.nn as nn
import torch.optim as optim
from skimage import io
from torch.utils.data import DataLoader
from .utils import net2RGB
from .colnet import ColNet
from .dataset import ImagesDateset
import matplotlib.pyplot as plt

class Train:
    def __init__(self,batch_size,epochs,img_dir_train, img_dir_val,img_dir_test,start_epoch=0,learning_rate=0.0001,
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
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.start_epoch = start_epoch

        self.img_dir_val = img_dir_val    
        self.devset = ImagesDateset(self.img_dir_val)
        self.devloader = DataLoader(self.devset, batch_size=self.BATCH_SIZE,
                                    shuffle=False, num_workers=num_workers)

        self.img_dir_train = img_dir_train
        self.trainset = ImagesDateset(self.img_dir_train)
        self.trainloader = DataLoader(self.trainset, batch_size=self.BATCH_SIZE, 
                                      shuffle=True, num_workers=num_workers)
        self.img_dir_test = img_dir_test
        self.testset = ImagesDateset(self.img_dir_test, testing=True)
        self.testloader = DataLoader(self.testset, batch_size=self.BATCH_SIZE,
                                     shuffle=False, num_workers=num_workers)
        
        self.loss_history = { "train": [], "val":[] }
        self.mse = nn.MSELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        self.net = ColNet(num_classes=self.num_classes)
        self.net.to(self.device)
        print("Using {}\n".format(self.device))
        if model_checkpoint:
            self.load_checkpoint(model_checkpoint)
        
        self.current_model_name = model_checkpoint
        self.best_val_loss = float("inf")
        self.best_model_dir = os.path.join(self.models_dir, 'colnet-the-best.pt')
        
        self.img_out_dir = img_out_dir
        self.models_dir = models_dir
    
    def loss(self,class_target,class_out,col_target,col_out):
        return self.mse(col_target,col_out) + self.ce(class_out,class_target)/300
    
    def train(self,epoch):
        epoch_loss = 0
        self.net.train()

        for idx in range(len(self.trainloader)):
            L, ab, labels = self.trainloader[idx]
            L = L.to(self.device)
            ab = ab.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            ab_out, labels_out = self.net(L)

            loss = self.loss(ab, ab_out, labels, labels_out)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            print('[Epoch ' + str(epoch+1) + '/' + str(self.EPOCHS) + ' | Batch: ' + str(idx +1) + '/' + str(len(self.trainloader)) + ']' + ' loss: ' + str(loss.item()))
        epoch_loss /= len(self.trainloader)
        self.loss_history['train'].append(epoch_loss)

        print("Epoch loss: ",epoch_loss)
    
    def validate(self,epoch):
        dev_loss = 0
        self.net.eval()
        with torch.no_grad():
            for idx in range(len(self.devloader)):
                L_dev, ab_dev, labels_dev = self.devloader[idx]
                L_dev = L_dev.to(self.device)
                ab_dev = ab_dev.to(self.device)
                labels_dev = labels_dev.to(self.device)
                ab_dev_out,labels_dev_out = self.net(L_dev)

                dev_batch_loss = self.loss(ab_dev,ab_dev_out,labels_dev,labels_dev_out)
                dev_loss += dev_batch_loss.item()
                print('[Validation] [Batch ' + str(idx +1) + '/' + str(len(self.devloader)) + ']' + ' dev loss: ' + str(dev_batch_loss))        
            dev_loss /= len(self.devloader)
            self.loss_history['val'].append(dev_loss)
            print("Dev loss ",dev_loss)
    
    def test(self,model_dir):
        if model_dir is None:
            model_dir = self.current_model_name

            if os.path.isfile(self.best_model_dir):
                model_dir = self.best_model_dir
        
        self.load_checkpoint(model_dir)
        self.net.to(self.device)
        self.net.eval()

    def save_checkpoint(self,epoch):
        full_path = os.path.join(self.models_dir,"colnet{}-{}.pt".format(time.strftime("%y%m%d-%H-%M-%S"), epoch))

        torch.save({
            'epoch': epoch,'model_state_dict': self.net.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.loss_history,'classes': self.classes}, full_path)

        self.current_model_name = full_path
        print('\nSaved the model to', full_path)

        if (self.loss_history['val'][-1] < self.best_val_loss):
            self.best_val_loss = self.loss_history['val'][-1]
            shutil.copy(full_path, self.best_model_dir)
            print("Saved the best model on epoch: ",epoch+1,"\n")

    def load_checkpoint(self,model_checkpoint):
        print("Resuming training of: " + model_checkpoint)
        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.loss_history = checkpoint['losses']
        self.current_model_name = model_checkpoint
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1 
        
    def plot_losses(self):
        print("Plotting the Model training losses")
        traininglosss = self.loss_history['train']
        validationloss = self.loss_history['val']
        trainlossplot = plt.plot(traininglosss)
        plt.ylabel('Epoch Loss')
        plt.ylabel('Epoch ')
        plt.savefig('trainlossplot.png')        

    def run(self):
        for epoch in range(self.start_epoch, self.EPOCHS):
            print("{2}\nEpoch {0} / {1}\n{2}".format(epoch + 1, self.EPOCHS, '-'*47))
            self.train(epoch)
            self.validate(epoch)
            self.save_checkpoint(epoch)
        print('\nFinished Training.\n')


if __name__ == "__main__":
    print("Training model")
            
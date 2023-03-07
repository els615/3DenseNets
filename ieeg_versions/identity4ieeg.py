# 6_10_retrain, dense_id_train_SUBSET_6_10_21.py
# SUBSET training
# Make a deep ReLU network and train it to label identity images
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import shelve
import math
#
device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
#
class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=7,stride=1,padding=3,bias=False)
        self.conv1_drop = nn.Dropout2d(p=0.5)
    def forward(self, x):
        out = self.conv1_drop(F.relu(self.conv1(self.bn1(x.float()))))
        out = torch.cat((x, out), 1)
        return out

#
class Transition(nn.Module):
    def __init__(self,nChannels,nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=7,stride=1,padding=3,bias=False)
        self.conv1_drop = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self,x):
        out = self.pool(self.conv1_drop(F.relu(self.conv1(self.bn1(x.float())))))
        return out

#
class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses):
        super(DenseNet, self).__init__()
        nDenseBlocks = (depth-4) // 3
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=7, padding=3,bias=True)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        #
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        #
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        #
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
#
    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
#
    def forward(self, x):
        out = self.conv1(x.float())
        x1 = out.clone().detach()
        out = self.trans1(self.dense1(out.float()))
        x2 = out.clone().detach()
        out = self.trans2(self.dense2(out.float()))
        x3 = out.clone().detach()
        out = self.dense3(out.float())
        x4 = out.clone().detach()
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out.float())), 8))
        out = F.log_softmax(self.fc(out.float())) # added dim = 1
        return out, x1, x2, x3, x4

#
#growthRate=16
#depth=13
#reduction=0.5
#nClasses=7
net = DenseNet(32, 13, 0.5, 1503)
net.cuda()
#
# Import data text file and split data 
import os
import glob
from random import *
import csv
#from sklearn.model_selection import train_test_split
import pandas as pd
df_train = pd.read_pickle('/gsfs0/data/schwarex/neuralNetworks/identity/datasets/train_SUBSET.pkl')
#df_train = pd.read_pickle('/mmfs1/data/schwarex/neuralNetworks/identity/datasets/train_SUBSET.pkl')
df_test = pd.read_pickle('/gsfs0/data/schwarex/neuralNetworks/identity/datasets/test_SUBSET.pkl')
#df_test = pd.read_pickle('/mmfs1/data/schwarex/neuralNetworks/identity/datasets/test_SUBSET.pkl')
#
# Implement the data loader.
from torch.utils.data import Dataset, DataLoader
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
#
class celebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, t_set, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataInfo_temp = t_set
        self.data = dataInfo_temp
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_name =  self.data.iloc[idx,0]     
        image = Image.open(img_name)
        label = self.data.iloc[idx,3] #identity labels
        index = self.data.iloc[idx,2] #index labels to pair image with extracted features
        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(178),
            transforms.Resize(48), #check resize/crop images are 218*178
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
                                            ])
        image = data_transforms(image)
        sample = {'image': image, 'label': label, 'index':index}
        if self.transform:
            sample = self.transform(sample)
        return sample

# Create loader
celebA_train_transformed = celebADataset(t_set=df_train,
                                        transform=None
                                        )
#
trainloader = DataLoader(celebA_train_transformed, batch_size=64,
                        shuffle=True, num_workers=2,pin_memory=True)
#
# Train the network
import torch.optim as optim
from torch.autograd import Variable
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9) # for 0.5 dropout
# optimizer = optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
#
def save_model(net,optim,ckpt_fname):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)
#
def adjust_learning_rate(optimizer, epoch):
    lr = 0.1*(0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#
net.train()
save_freq = 50
save_dir = '/gsfs0/data/schwarex/neuralNetworks/identity/denseNet/6_10_retrain' #save checkpoints here
#save_dir = '/mmfs1/data/schwarex/neuralNetworks/identity/densenet/6_10_retrain'
loss_memory = []
for epoch in range(0,201):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        adjust_learning_rate(optimizer,epoch)
        # get the inputs
        images = data['image']
        labels = data['label']
        indices = data['index']
        tmp = []
        tmp = torch.squeeze(labels.long())
        tmpInd = []
        tmpInd = torch.squeeze(indices.long())
        images, labels, indices = images.cuda(), tmp.cuda(), tmpInd.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(images) 
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data
        if i % 50 == 49:    # print every 100 mini-batches
            loss_memory.append(running_loss/100)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    if epoch % save_freq == 0: 
        save_model(net, optimizer, os.path.join(save_dir, '%03d.ckpt' % epoch))

#
# Make the dataloader for testing
celebA_train_transformed = None
trainloader = None
celebA_test_transformed = celebADataset(t_set=df_test,
                                        transform=None
                                        )
testloader = DataLoader(celebA_test_transformed, batch_size=64,
                        shuffle=True, num_workers=2,pin_memory=True)
# Test
net.eval()
score = []
image_pred_final = None
for i, data in enumerate(testloader):
    # get the inputs
    images = data['image']
    labels = data['label']
    indices = data['index']
    tmp = []
    tmp = torch.squeeze(labels.long())
    tmpInd = []
    tmpInd = torch.squeeze(indices.long())
    images, labels, indices = images.cuda(), tmp.cuda(), tmpInd.cuda()  
    # forward + backward + optimize
    all_outputs = net(images)
    output_final = all_outputs[0].cpu().data.detach().numpy()
    output_argmax = np.argmax(output_final,axis=1)
    labels_numpy = labels.cpu().data.numpy()
    score = np.concatenate((score,(labels_numpy==output_argmax).astype(int)),axis=0)
meanAccuracy = sum(score)/len(score)
print(meanAccuracy)
#
# Save
d = shelve.open(save_dir+'/results')
d['loss'] = loss_memory
d['accuracy'] = meanAccuracy
print(meanAccuracy)

d.close()


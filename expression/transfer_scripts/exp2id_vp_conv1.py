import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import shelve
import math
from statistics import mean

device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

class Transfer(nn.Module):
    def __init__(self, nClasses):
        super(Transfer, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1.requires_grad=False
        self.fc = nn.Linear(36*64, nClasses) #conv 1 147456 # lay1 46080, lay2 12672, lay 3 26496.  368640 101376. 
    def forward(self,x):
        #x = x.view(-1,2048)
        #print(x.shape)
        #x = x.view(x.size(0),-1)
        x = torch.squeeze(x)
        x = F.avg_pool2d(F.relu(self.bn1(x.float())), 8)
        x = x.view(-1,6*6*64)
        #print(x.shape)
        x = F.log_softmax(self.fc(x.float()))
        return x

net = Transfer(7)
net.cuda()

import os
import glob
from random import *

# Implement the data loader.
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from torchvision import transforms, utils

## take all features, put in df with labels
class KDEFDataset(Dataset):
    def __init__(self, dataframe, usage, transform=None):
        dataInfo_temp = dataframe
        # take only the part of the file that's relevant for the current usage
        dataInfo_temp = dataInfo_temp[dataInfo_temp.usage == usage]
        self.dataInfo = dataInfo_temp.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.dataInfo)
    def __getitem__(self, idx):
        image = self.dataInfo.iloc[idx, 6] #6 conv1 
        label = self.dataInfo.iloc[idx, 5] #exp label
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image)
        #image = torch.unsqueeze(image,0)
        return {'image': torch.unsqueeze(image,0), # add 
                'label': torch.from_numpy(np.array([label]))}

import torch.optim as optim
from torch.autograd import Variable
criterion = nn.CrossEntropyLoss()
angles = ['S', 'HL', 'HR']
accs_all = list()
accs_angle_mean = list()
for i, view in enumerate(angles):
    for rep in range(1,11):
        accs_angle = list()
        path = '/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/expression/kdef/kdef_features_merged_%s_%d.pkl' % (view, rep)
        dataPath = pd.read_pickle(path)
        net = Transfer(7)
        net.cuda()
        KDEF_train_transformed = KDEFDataset(dataframe=dataPath, 
                                                   usage="Training",
                                                   transform=transforms.Compose([
                                                       ToTensor()
                                                   ]))
        trainloader = DataLoader(KDEF_train_transformed, batch_size=32,
                                shuffle=True, num_workers=2,pin_memory=False)
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,weight_decay=0.000)
        net.train()
        save_freq = 50
        #os.mkdir(save_dir)
        loss_memory = []
        for epoch in range(100):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                # get the inputs
                images = data['image']
                labels = data['label']
                #tmp = []
                #tmp = torch.squeeze(labels.long())
                images, labels = images.cuda(), labels.long().cuda()
                #images, labels = Variable(images.cuda()),  Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = net(images) 
                #loss = criterion(output, labels)
                loss = criterion(output, torch.max(labels, 1)[0])
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.data
                if i % 50 == 49:    # print every 100 mini-batches
                    loss_memory.append(running_loss/100)
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0
        KDEF_train_transformed = None
        trainloader = None
        KDEF_test_transformed = KDEFDataset(dataframe=dataPath,
                                                   usage="PublicTest",
                                                   transform=transforms.Compose([
                                                       ToTensor()
                                                   ]))
        testloader = DataLoader(KDEF_test_transformed, batch_size=32,
                                shuffle=True, num_workers=2,pin_memory=True)
        net.eval()
        image_pred_final = None
        score = []
        for i, data in enumerate(testloader):
            # get the inputs
            images = data['image']
            labels = data['label']
            tmp = []
            tmp = torch.squeeze(labels.long())
            images, labels = images.cuda(), tmp.cuda()
            # forward + backward + optimize
            output = net(images)
            output = output.cpu().data.detach().numpy()
            output_argmax = np.argmax(output,axis=1)
            labels_numpy = labels.cpu().data.numpy()
            score = np.concatenate((score,(labels_numpy==output_argmax).astype(int)),axis=0)
            try:
                image_pred_final = np.concatenate((image_pred_final, output))
            except:
                image_pred_final = output
        meanAccuracy = sum(score)/len(score)
        print(meanAccuracy)
        accs_all.append(meanAccuracy)
        accs_angle.append(meanAccuracy)
    accs_angle_mean.append(mean(accs_angle))

print(mean(accs_all))
print(accs_angle_mean)
print(accs_all)

from statistics import mean
from scipy.stats import sem
avg_10 = list()
for val in range(0,10):
    avg_1 = list()
    avg_1.append(accs_all[val])
    avg_1.append(accs_all[val+10])
    avg_1.append(accs_all[val+20])
    avg_10.append(mean(avg_1))

print(avg_10)
print(mean(avg_10))
print(sem(avg_10))


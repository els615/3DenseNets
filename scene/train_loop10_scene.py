import pandas as pd 
import PIL
from PIL.ImageOps import grayscale 
import numpy as np
import os
dataset_path = '/mmfs1/data/schwarex/neuralNetworks/datasets/UCMerced_LandUse/Images'

pixel_vals = []
category_labels = []
usage = []

idx = 1
for root, subdirectories, files in os.walk(dataset_path):
    for file in files:
        image_file = os.path.join(root, file)
        im = PIL.Image.open(image_file)
        im = grayscale(im)
        im = im.resize((48,48))
        pixel_vals.append(np.asarray(im))
        category_labels.append(int(idx/100))
        if idx%10 == 0:
          usage.append('Testing')
        else:
          usage.append('Training')
        idx+=1


df = pd.DataFrame({'image':pixel_vals, 'label':category_labels, 'usage':usage})

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import shelve
import math

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,stride=1,padding=1,bias=False)
        self.conv1_drop = nn.Dropout2d(p=0.1)
    def forward(self, x):
        out = self.conv1_drop(F.relu(self.conv1(self.bn1(x.float()))))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self,nChannels,nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=3,stride=1,padding=1,bias=False)
        self.conv1_drop = nn.Dropout2d(p=0.1)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self,x):
        out = self.pool(self.conv1_drop(F.relu(self.conv1(self.bn1(x.float())))))
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses):
        super(DenseNet, self).__init__()
        nDenseBlocks = (depth-4) // 3
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,bias=True)
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
        out = self.trans1(self.dense1(out.float()))
        out = self.trans2(self.dense2(out.float()))
        out = self.dense3(out.float())
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out.float())), 8))
        out = F.log_softmax(self.fc(out.float()))
        return out

#growthRate=16
#depth=13
#reduction=0.5
#nClasses=7
net = DenseNet(32, 13, 0.5, 21)
net.cuda()

import os
import glob
from random import *
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

class MercedDataset(Dataset):
  def __init__(self, dataInfo_temp, usage, transform=None):
    self.dataInfo = (dataInfo_temp[dataInfo_temp.usage == usage]).reset_index(drop=True)
    self.transform = transform
  def __len__(self):
      return len(self.dataInfo)
  def __getitem__(self, idx):
      #image = np.reshape(np.array(np.fromstring(self.dataInfo.iloc[idx, 1], np.uint8, sep=' ')),(48,48)) # reshape image
      image = self.dataInfo['image'][idx]
      #label = self.dataInfo.iloc[idx, 0]   
      label = self.dataInfo['label'][idx]
      sample = {'image': image, 'label': label}
      if self.transform:
          sample = self.transform(sample)
      return sample


class Resize(object):
    """Resize the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, larger of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size , self.output_size* w / w
            else:
                new_h, new_w = self.output_size* h / w, self.output_size 
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        return {'image': img, 'label': label}

class Flip(object):
    """Translate randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.randint(0,1)>0:
            image = np.fliplr(image)
        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = torch.from_numpy(image)
        return {'image': torch.unsqueeze(image,0), # add 
                'label': torch.from_numpy(np.array([label]))}

def save_model(net,optim,ckpt_fname):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)

def adjust_learning_rate(optimizer, epoch):
    lr = 0.1*(0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#

import torch.optim as optim
from torch.autograd import Variable
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for rep in range(0,10):
    net = DenseNet(32, 13, 0.5, 21)
    net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # Create loader
    merced_train_transformed = MercedDataset(dataInfo_temp=df, usage='Training', transform=transforms.Compose([
                                                    Resize(48),
                                                    Flip(),
                                                    ToTensor()
                                               ]))
    trainloader = DataLoader(merced_train_transformed, batch_size=64,
                            shuffle=True, num_workers=2,pin_memory=True)
    #
    # Train the network
    net.train()
    save_freq = 50
    save_dir = '/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/scene'
    loss_memory = []
    for epoch in range(0,200):  # loop over the dataset multiple times
        running_loss = 0.0
        #count = 0
        #for name, param in net.named_parameters():
        #    if param.requires_grad and count == 0:
        #        print(name, param.data)
        #        count = count + 1
        for i, data in enumerate(trainloader):
            adjust_learning_rate(optimizer,epoch)
            # get the inputs
            images = data['image']
            labels = data['label']
            tmp = []
            tmp = torch.squeeze(labels.long())
            images, labels = images.cuda(), tmp.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(images) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data
            if i % 50 == 49:    # print every 100 mini-batches
                loss_memory.append(running_loss/100)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        if epoch % save_freq == save_freq-1: 
            save_model(net, optimizer, os.path.join(save_dir, '%03d_%d.ckpt' % (epoch, rep+1)))

    #
    # Make the dataloader for testing
    merced_train_transformed = None
    trainloader = None
    merced_test_transformed = MercedDataset(dataInfo_temp=df, usage='Testing', transform=transforms.Compose([
                                               Resize(48),
                                               Flip(),
                                               ToTensor(),
                                           ]))
    testloader = DataLoader(merced_test_transformed, batch_size=64,
                            shuffle=True, num_workers=2,pin_memory=True)
    # Test
    net.eval()
    score = []
    image_pred_final = None
    for i, data in enumerate(testloader):
        # get the inputs
        images = data['image']
        labels = data['label']
        tmp = []
        tmp = torch.squeeze(labels.long())
        images, labels= images.cuda(), tmp.cuda() 
        # forward + backward + optimize
        output = net(images)
        output_final = output.cpu().data.detach().numpy()
        output_argmax = np.argmax(output_final,axis=1)
        labels_numpy = labels.cpu().data.numpy()
        score = np.concatenate((score,(labels_numpy==output_argmax).astype(int)),axis=0)
    meanAccuracy = sum(score)/len(score)
    print(meanAccuracy)
    #
    # Save
    result_dir = save_dir + '/results_%d' % (rep+1)
    d = shelve.open(result_dir)
    d['loss'] = loss_memory
    d['accuracy'] = meanAccuracy
    print(meanAccuracy)

    d.close()




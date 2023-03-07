# train using fer2013 to label expression
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import shelve
import math

device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

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

#growthRate=16
#depth=13
#reduction=0.5
#nClasses=7
net = DenseNet(32, 13, 0.5, 7)
net.cuda()

# Import data path
import os
import glob
from random import *
import csv
dataPath = '/gsfs0/data/schwarex/ecog_project/data/FER2013/fer2013.csv'

# Implement the data loader.
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

class Fer2013Dataset(Dataset):
    """FER 2013 dataset."""
    def __init__(self, csv_file, usage, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataInfo_temp = pd.read_csv(csv_file)
	# take only the part of the file that's relevant for the current usage
        self.dataInfo = (dataInfo_temp[dataInfo_temp.Usage == usage]).reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.dataInfo)
    def __getitem__(self, idx):
        image = np.reshape(np.array(np.fromstring(self.dataInfo.iloc[idx, 1], np.uint8, sep=' ')),(48,48)) # reshape image
        label = self.dataInfo.iloc[idx, 0]   
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

# Make normalized loader
fer2013_train_transformed = Fer2013Dataset(csv_file=dataPath,
                                           usage='Training',
                                           transform=transforms.Compose([
                                               Resize(48),
                                               Flip(),
                                               ToTensor(),
                                           ]))

trainloader = DataLoader(fer2013_train_transformed, batch_size=64,
                        shuffle=True, num_workers=2,pin_memory=True)

# Train the network
import torch.optim as optim
from torch.autograd import Variable
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # typical
# optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9) # for 0.5 dropout
# optimizer = optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0)

def save_model(net,optim,ckpt_fname):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)

net.train()
save_freq = 50
save_dir = '/gsfs0/data/schwarex/ecog_project/data/FER2013/'
loss_memory = []
for epoch in range(201):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
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

# Make the dataloader for testing
fer2013_train_transformed = None
trainloader = None
fer2013_test_transformed = Fer2013Dataset(csv_file=dataPath,
                                            usage='PublicTest',
                                            transform=transforms.Compose([
                                               Resize(48),
                                               Flip(),
                                               ToTensor(),
                                           ]))
testloader = DataLoader(fer2013_test_transformed, batch_size=64,
                        shuffle=True, num_workers=2,pin_memory=True)
# Test
net.eval()
score = []
image_pred_final = None
image_pred_layer1 = None
image_pred_layer2 = None
image_pred_layer3 = None
image_pred_layer4 = None

for i, data in enumerate(testloader):
    # get the inputs
    images = data['image']
    labels = data['label']
    tmp = []
    tmp = torch.squeeze(labels.long())
    images, labels = images.cuda(), tmp.cuda()
    
    # forward + backward + optimize
    all_outputs = net(images)
    
    output_final = all_outputs[0].cpu().data.detach().numpy()
    output_layer1 = all_outputs[1].cpu().data.detach().numpy()
    output_layer2 = all_outputs[2].cpu().data.detach().numpy()
    output_layer3 = all_outputs[3].cpu().data.detach().numpy()
    output_layer4 = all_outputs[4].cpu().data.detach().numpy()
    
    output_argmax = np.argmax(output_final,axis=1)
    labels_numpy = labels.cpu().data.numpy()
    score = np.concatenate((score,(labels_numpy==output_argmax).astype(int)),axis=0)

    
    try:
        image_pred_final = np.concatenate((image_pred_final, output_final))
        image_pred_layer1 = np.concatenate((image_pred_layer1, output_layer1))
        image_pred_layer2 = np.concatenate((image_pred_layer2, output_layer2))
        image_pred_layer3 = np.concatenate((image_pred_layer3, output_layer3))
        image_pred_layer4 = np.concatenate((image_pred_layer4, output_layer4))
    except ValueError:
        image_pred_final = output_final
        image_pred_layer1 = output_layer1
        image_pred_layer2 = output_layer2
        image_pred_layer3 = output_layer3
        image_pred_layer4 = output_layer4

meanAccuracy = sum(score)/len(score)


# Save
d = shelve.open(save_dir+'/results')
d['loss'] = loss_memory
d['accuracy'] = meanAccuracy

# save extracted arrays
d['final_output'] = image_pred_final
d['layer1_output'] = image_pred_layer1
d['layer2_output'] = image_pred_layer2
d['layer3_output'] = image_pred_layer3
d['layer4_output'] = image_pred_layer4

d.close()


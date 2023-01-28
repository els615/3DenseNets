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
#
device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')
save_dir = '/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef'
#
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
        x1 = out.clone().detach()
        #out = self.trans1(self.dense1(out.float()))
        out = self.dense1(out.float())
        x2 = out.clone().detach()
        out = self.trans1(out.float())
        #out = self.trans2(self.dense2(out.float()))
        out = self.dense2(out.float())
        x3 = out.clone().detach()
        out = self.trans2(out.float())
        out = self.dense3(out.float())
        x4 = out.clone().detach()
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out.float())), 8))
        #x4 = out.clone().detach()
        # x5 = out.clone().detach()
        out = self.fc(out.float()) # added dim = 1
        x5 = out.clone().detach()
        out = F.log_softmax(out)
        return out, x1, x2, x3, x4, x5

import os
import glob
from random import *
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_pickle('/mmfs1/data/schwarex/neuralNetworks/datasets/KDEF_3views.pkl') # will not drop by usage

#
# Implement the data loader.
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms, utils
from PIL import Image
#
class KDEFDataset(Dataset):
    """KDEF dataset."""
    def __init__(self, t_set, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    # take only the part of the file that's relevant for the current usage
        self.dataAll = t_set
        #self.dataManyImages = (dataInfo_temp[dataInfo_temp['numImages']>1]).reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.dataAll)
    def __getitem__(self, idx):
        img_name =  self.dataAll.iloc[idx,3]     
        image = Image.open(img_name)
        label = self.dataAll.iloc[idx,0] #identity labels
        index = self.dataAll.iloc[idx,4] #index labels to pair image with extracted features
        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(388),
            transforms.Resize(48), 
        #    transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
                                            ])
        image = data_transforms(image)
        sample = {'image': image, 'label': label, 'index':index}
        if self.transform:
            sample = self.transform(sample)
        return sample

#
for rep in range(1,11):
    checkpoint_file = '/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity_v2/199_%d.ckpt' % rep
    net = DenseNet(32, 13, 0.5, 1503)
    savedModel=torch.load(checkpoint_file)
    state_dict=savedModel['state_dict']
    optimizer=savedModel['optimizer']
    epoch=savedModel['epoch']
    net.load_state_dict(state_dict)
    net.to(device)
    for name, param in net.named_parameters():
        param.requires_grad = False
    # Create loader
    KDEF_test_transformed = KDEFDataset(t_set=df,
                                            transform=None
                                            )
    testloader = DataLoader(KDEF_test_transformed, batch_size=64,
                            shuffle=True, num_workers=2,pin_memory=True)
    net.eval()
    score = []
    image_pred_final = None
    image_pred_layer1 = None
    image_pred_layer2 = None
    image_pred_layer3 = None
    image_pred_layer4 = None
    image_pred_layer5 = None
    idx_arr = None
    count = 0
    for i, data in enumerate(testloader):
        # get the inputs
        images = data['image']
        labels = data['label']
        indices = data['index']
        tmp = []
        tmp = torch.squeeze(labels.long())
        tmpInd = []
        tmpInd = torch.squeeze(indices.long())
        images, labels, indices = images.to(device), tmp.to(device), tmpInd.to(device)
        # forward + backward + optimize
        all_outputs = net(images)
        output_final = all_outputs[0].cpu().data.detach().numpy()
        output_layer1 = all_outputs[1].cpu().data.detach().numpy()
        output_layer2 = all_outputs[2].cpu().data.detach().numpy()
        output_layer3 = all_outputs[3].cpu().data.detach().numpy()
        output_layer4 = all_outputs[4].cpu().data.detach().numpy()
        output_layer5 = all_outputs[5].cpu().data.detach().numpy()
        output_argmax = np.argmax(output_final)
        labels_numpy = labels.cpu().data.numpy()
        indices = indices.cpu().data.numpy()
        score = np.concatenate((score,(labels_numpy==output_argmax).astype(int)),axis=0)
        try:
            image_pred_final = np.concatenate((image_pred_final, output_final))
            image_pred_layer1 = np.concatenate((image_pred_layer1, output_layer1))
            image_pred_layer2 = np.concatenate((image_pred_layer2, output_layer2))
            image_pred_layer3 = np.concatenate((image_pred_layer3, output_layer3))
            image_pred_layer4 = np.concatenate((image_pred_layer4, output_layer4))
            image_pred_layer5 = np.concatenate((image_pred_layer5, output_layer5))
            idx_arr = np.concatenate((idx_arr, indices))
        except ValueError:
            image_pred_final = output_final
            image_pred_layer1 = output_layer1
            image_pred_layer2 = output_layer2
            image_pred_layer3 = output_layer3
            image_pred_layer4 = output_layer4
            image_pred_layer5 = output_layer5
            idx_arr = indices
    list_of_tuples = list(zip(image_pred_final, image_pred_layer1, image_pred_layer2, image_pred_layer3, image_pred_layer4, image_pred_layer5, idx_arr)) 
    df2 = pd.DataFrame(list_of_tuples, columns = ['final_output', 'conv1', 'lay1', 'lay2', 'lay3', '2nd_last', 'idx'])
    df2.to_pickle(save_dir+'/kdef_features_%d.pkl' % rep) 
    # merge
    new_df2 = pd.merge(df,df2,on=['idx'])
    new_df2.to_pickle(save_dir+'/kdef_features_merged_%d.pkl' % rep)

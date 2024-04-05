"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = F.cross_entropy(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))



class CIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)                                # Batch Normalization Layer for increasing accuracy
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)                                # Batch Normalization Layer for increasing accuracy
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, OutputSize)
        self.InputSize = InputSize
        self.OutputSize = OutputSize

    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        out = F.relu(self.conv1(xb))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 32 * 8 * 8)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
class DenseNet_Model(ImageClassificationBase):
    def __init__(self, growth_rate=12, block_config=[6, 12, 24], num_classes=10):
        super(DenseNet_Model, self).__init__()
        # Your code here
        self.growth_rate = growth_rate 
        self.block_config = block_config
        self.num_classes = num_classes
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, 2 * growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(2 * growth_rate))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        num_features = 2 * growth_rate
        
        #Dense Block 1
        for i in range(block_config[0]):
            self.features.add_module('denseblock1_layer{}'.format(i + 1), _DenseLayer(num_input_features=num_features, growth_rate=growth_rate))
            num_features += growth_rate
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('pool5', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        self.features.add_module('conv6', nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1))  # Add a convolutional layer
        self.features.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))  # Add AdaptiveAvgPool2d layer
        self.classifier = nn.Linear(num_features, num_classes)  # Adjust input size of the linear layer
        self.InputSize = 32
        self.OutputSize = 10

        #Dense Block 2
        for i in range(block_config[1]):
            self.features.add_module('denseblock2_layer{}'.format(i + 1), _DenseLayer(num_input_features=num_features, growth_rate=growth_rate))
            num_features += growth_rate
        self.features.add_module('norm6', nn.BatchNorm2d(num_features))
        self.features.add_module('relu6', nn.ReLU(inplace=True))
        self.features.add_module('pool6', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        self.classifier = nn.Linear(num_features, num_classes)
        self.InputSize = 32
        self.OutputSize = 10

    def forward(self, xb):
        out = self.features(xb)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0):
        super(_DenseLayer, self).__init__()
        # Your code here
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate
        
    def forward(self, xb):
        # Your code here
        new_features = super(_DenseLayer, self).forward(xb)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([xb, new_features], 1)
    
class ResNet_Model(ImageClassificationBase):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Model, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._generate_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._generate_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._generate_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.InputSize = 32
        self.OutputSize = 10

    def _generate_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion 
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out) 
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, groups=32, width_per_group=4):
        super(ResNeXtBottleneck, self).__init__()
        width = groups * width_per_group
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * expansion)
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
# class ResNext_Model():
#     return ResNet_Model(ResNeXtBottleneck, [3, 4, 6, 3])
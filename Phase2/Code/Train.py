#!/usr/bin/env python3

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

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)



from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random
import os
import random
from torchvision.transforms import ToTensor
import argparse
from tqdm import tqdm
from Network.Network import CIFAR10Model, DenseNet_Model, ResNet_Model, ResNext_Model
from Misc.MiscUtils import *
import sys
from Misc.DataUtils import SetupAll
import matplotlib.pyplot as plt
from torchsummary import summary
    
def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    ImageSize = [32, 32, 3]

    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1) 
        
        ImageNum += 1
        
          ##########################################################
          # Add any standardization or data augmentation here!
          ##########################################################
        # I1 = TrainSet[RandIdx]
        # I1 = I1 / 255  # Scale to [0,1]
        # I1 = (I1 - 0.5) * 2  # Scale to [-1,1]
        I1, Label = TrainSet[RandIdx]

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    
# def load_dataset():
#         # Replace with your actual function to load the dataset
#         # Load the dataset
#         TrainSet = torchvision.datasets.CIFAR10(root= '.\CIFAR10\Train\ ', train=True,
#                                         download=False, transform=ToTensor())
#         return TrainSet 


def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, LogsPath):
   
    """
    Inputs: 
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    
    
    # Initialize the model
    #model = CIFAR10Model(InputSize=3*32*32,OutputSize=10)
    #model  = DenseNet_Model()
    model = ResNet_Model(InputSize=3*32*32,OutputSize=10)
    #model = ResNext_Model(InputSize=3*32*32,OutputSize=10)
    summary(model, (3, 32, 32))

    # Initialize the optimizer
    Optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=(1e-08), weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=30, gamma=0.1)
    # Initialize the Tensorboard writer
    Writer = SummaryWriter(log_dir=LogsPath)

    # # Load the dataset
    # TrainSet = load_dataset()  # replace with your actual function to load the dataset
    
    train_accuracy = []

    if LatestFile is not None:                                              # Load the model from latest checkpoint
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')      # Load checkpoint                  
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))   # Extract the epoch number from the file name
        model.load_state_dict(CheckPoint['model_state_dict'])              # Load Model State Dict
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')     
    else:
        StartEpoch = 0                                                # Start from Epoch 0 if there is no checkpoint
        print('New model initialized....')
    
    correct_predictions = 0
    total_samples = 0
    # Start Training
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):                        # Loop over Epochs
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)  # Number of iterations per epoch, here one iteration is 1 mini-batch
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):           #  Loop over Mini-Batches for every Epoch 
            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize) # Generate Batch
            
            # Predict output with forward pass
            LossThisBatch = model.training_step(Batch)  # Loss for this batch

            Optimizer.zero_grad()  # Clear gradients from the previous iteration
            LossThisBatch.backward() # Backpropagate the loss
            Optimizer.step() # Update the weights
            
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                
                torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
                print('\n' + SaveName + ' Model Saved...')

            result = model.validation_step(Batch)
            model.epoch_end(Epochs*NumIterationsPerEpoch + PerEpochCounter, result)
            # Tensorboard
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            Writer.add_scalar('Accuracy', result["acc"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()         
        scheduler.step()
        
        # Compute the accuracy
        # for data, labels in train_loader:  # assuming you have a DataLoader named train_loader
        #     data, labels = data.to(device), labels.to(device)  # assuming you have a device object for GPU/CPU
        #     outputs = model(data)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total_samples += labels.size(0)
        #     correct_predictions += (predicted == labels).sum().item()

        # for data, labels in train_loader:  # assuming you have a DataLoader named train_loader
        #     data, labels = data.to(device), labels.to(device)  # assuming you have a device object for GPU/CPU
        #     outputs = model(data)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total_samples += labels.size(0)
        #     correct_predictions += (predicted == labels).sum().item()

        # accuracy = correct_predictions / total_samples
        # Writer.add_scalar('Train Accuracy', accuracy, Epochs)
        # Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=20, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    
    # # Setup all needed parameters including file reading
    # TrainSet = load_dataset()  # Define TrainSet variable

    x, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath='./data/', CheckPointPath=CheckPointPath)


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, LogsPath)
    

    
if __name__ == '__main__':
    main()
 

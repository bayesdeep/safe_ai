import sys, os
import torch
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils.dataloader import A2d2dataset
from data_utils.utils import train_val_test
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np  
sys.path.append('network')
import network



def train_step(model, train_dataloader, valid_dataloader,epochs, loss_fn, optimizer, device):
   
    """ performs training and validation.
        :param model: model to train.
        :param train_dataloader: a daatloder with training images and labels
        :param valid_dataloader :a daatloder with validation images and labels
        :param epochs (int): number of epochs to train the model.
        :param loss_fn: loss function to compute loss.
        :param optimizer: optimizer to perfrom weight updates.
        :param device: either cuda or cpu depending on the system.

        :return: train and validation loss
    """
   
    train_loss, valid_loss = [], []
    for epoch in range(epochs):
        train_running_loss = 0.0
        valid_running_loss = 0.0
        model.train()  # set model in training mode

        for idx, (images, labels) in enumerate(train_dataloader):
            images = images.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            predicted = model(images)
            loss = loss_fn(torch.sigmoid(predicted), labels) #compute loss
            loss.backward()  #backpropogation
            optimizer.step() # update the model parameters
            train_running_loss += loss.item()
        epoch_train_loss = train_running_loss/len(train_dataloader)
        train_loss.append(epoch_train_loss)
        print('-----training done, valdiating-----')
        model.eval()
        with torch.no_grad():
            for idx, (val_images, val_labels) in enumerate(valid_dataloader):
                val_images = val_images.float().to(device)
                val_labels = val_labels.float().to(device)
                predicted = model(val_images)
                loss = loss_fn(torch.sigmoid(predicted), val_labels)
                valid_running_loss += loss.item()
            epoch_valid_loss = valid_running_loss/len(valid_dataloader)
            valid_loss.append(epoch_valid_loss)
        print('epoch: {}/{}, traning_loss :{:3f}, validation_loss :{:3f}'.format(epoch, epochs, epoch_train_loss, epoch_valid_loss))
        
    torch.save(model.state_dict(), 'new_model_weights_dropout_completedata.pth')

    return train_loss, valid_loss

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize ', type= int, default= 16, help = 'define the batchsize')
    parser.add_argument('--epochs', type= int, default= 25, help = 'define the number of epochs to train')
    parser.add_argument('--lr', type = float, default= 1e-3, help = 'specify the learning rate')
    args = parser.parse_args()

    path_to_data = 'a2d2_compressed'
    transform = transforms.Compose([transforms.ToTensor()]) #some transformation
    batch =16  
    a2d2_dataset = A2d2dataset(root = 'a2d2_compressed', image_folder= 'images_traffic', labels_folder= 'binary_labels_traffic',transforms= transform)
    split_dataset = train_val_test(dataset= a2d2_dataset)
    train_dataloader = DataLoader(split_dataset['train'], batch_size =batch, shuffle= True)
    valid_dataloader = DataLoader(split_dataset['val'], batch_size =  batch, shuffle= False)
    # test_dataloader = DataLoader(split_dataset['test'], batch_size =  args.batch_size, shuffle= False)

    ##########################   DeeplabV3plus model adopted from https://github.com/VainF/DeepLabV3Plus-Pytorch #########################################

    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)
    model.load_state_dict(torch.load('best_deeplabv3plus_mobilenet_cityscapes_os16.pth')["model_state"])  ### adopted from the github repository (https://github.com/VainF/DeepLabV3Plus-Pytorch)

    for params in model.backbone.parameters():
        params.requires_grad = False
        
    model.classifier.project = model.classifier.project.append(nn.Dropout(p = 0.3,inplace= False))
    model.classifier.classifier[-1] =  torch.nn.Conv2d(in_channels= 256, out_channels= 1, kernel_size= (1, 1), stride= (1,1))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


    training_loss, validation_loss = train_step(model, train_dataloader, valid_dataloader, args.epochs, loss_function, optimizer, device)
    plt.plot(range(args.epochs), training_loss, label = 'Train loss')
    plt.plot(range(args.epochs), validation_loss, label = 'valistion loss')
    plt.title('loss curve')
    plt.legend()
    plt.savefig('training_losscurve.png', dpi = 200)
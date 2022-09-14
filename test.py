
import torch
import time
import matplotlib.pyplot as plt
from metrics import iou_metric, entropy
import numpy as np

def visualize(image, label, predicted, entropy_estimation,title, save_img= None):
    """to perfomm visualization on theinferred sample  
        :param images (tensor): image to predict
        :param lables (tensor): assosiated label
        :param predicted (tensor): sigmoid output
        :param entorpy_estimation(tensor): tensor with the entropy values
        :param save_img (str): provide a string to save the image
        :return: average_pred: average of all the forward passes.
                 entropy_ : estimated entopy of the average_prediction.
    """
    fig, ax = plt.subplots(1, 4, figsize = (40,50))
    if image.dim()==4:
        ax[0].imshow(image.squeeze(0).cpu().numpy().transpose(1,2,0))
        ax[0].set_title('camera_image')
        ax[1].imshow(label.squeeze(0).cpu().numpy().transpose(1,2,0))
        ax[1].set_title(title + 'Label')
        ax[2].imshow(predicted.squeeze(0).transpose(1,2,0))
        ax[2].set_title(title + 'predicted')
        ax[3].imshow(entropy_estimation.transpose(1,2,0), cmap= 'gist_heat')
        ax[3].set_title(title + 'predictive uncertainty')
        ax[3].grid()
    else:
        ax[0].imshow(image.cpu().numpy().transpose(1,2,0))
        ax[0].set_title('camera_image', fontsize=20)
        ax[1].imshow(label.cpu().numpy().transpose(1,2,0))
        ax[1].set_title('Label', fontsize=20)
        ax[2].imshow(predicted.transpose(1,2,0))
        ax[2].set_title(title + 'predicted', fontsize=20)
        ax[3].imshow(entropy_estimation.transpose(1,2,0))
        ax[3].set_title(title + 'predictive uncertainty', fontsize=20)
        ax[3].grid()
    if not save_img == None:
        plt.savefig(save_img)
        

def test_predictions(model, image, label, device):
    """This function performs inference on a single image 
        :param images (tensor): image to predict
        :param lables (tensor): assosiated label
        :param device (str): cuda or cpu depending on the device

        :return: average_pred: average of all the forward passes.
                 entropy_ : estimated entopy of the average_prediction.
    """
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        if image.dim() == 4:            
            images = image.float().to(device)
            labels = label.float().to(device)
            predicted = model(images)
            predicted = torch.sigmoid(predicted).cpu().numpy()
        else:
            images = image.unsqueeze(0).float().to(device)
            labels = label.unsqueeze(0).float().to(device)
            predicted = model(images)

            predicted = torch.sigmoid(predicted).cpu().numpy()
        entropy_ = entropy(predicted)

        iou = iou_metric(predicted, labels.cpu().numpy())
        
        visualize(images, labels, predicted, entropy_, title= 'baseline_', save_img= 'baseline_prediction.png')
        print('iou', iou)
        print('time taken:',time.time() -start_time)
    return predicted, entropy_

    
def enable_dropout(model):

    """ Function to enable the dropout layers during test-time 
        adopted from the discussion:  https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
             m.train()
             
def mcd_pred(model, images, labels,device, forward_passes = 5):

    """Function to perform Monte-carlo dropout during inference time.
        :param images (tensor): image to predict
        :param lables (tensor): assosiated label
        :param forward_passes (int): number of forward passes to perform

        :return: average_pred: average of all the forward passes.
                 entropy_ : estimated entopy of the average_prediction.
    """
    start_time = time.time()
    enable_dropout(model)
    mcd = []
    for  i in range(forward_passes):
        with torch.no_grad():
            if images.dim() == 4:            
                images = images.float().to(device)
                labels = labels.float().to(device)
                predicted = model(images)
                predicted = torch.sigmoid(predicted).cpu().numpy() 
                mcd.append(predicted)
            else:
                images = images.unsqueeze(0).float().to(device)
                labels = labels.unsqueeze(0).float().to(device)
                predicted = model(images)
                predicted = torch.sigmoid(predicted).cpu().numpy()  
                mcd.append(predicted)
    
    mcd = np.asarray(mcd)
    average_pred = np.mean(mcd, axis = 0)
    entropy_ = entropy(predicted)
    iou = iou_metric(predicted , labels.cpu().numpy()) 
    print('iou', iou)
    print('time taken:',time.time() -start_time)
    visualize(images, labels, predicted, entropy_,title= 'mc_dropout_', save_img= 'mcdropout_prediction.png') 
    return average_pred, entropy_



   
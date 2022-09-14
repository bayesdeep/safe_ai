# Safe AI Tech Challenge
Overview:
    pefromed a binary semantic segmentation of traffic lights using DeepLabv3plus model and evaluated the performance and model uncertainty on the clean and augmented test sampels, using the standard inference technique and MC-dropout method to attain the better uncertainty estimation.

Quick access:
    1. main.ipynb: This notebook is for the visualization, model training, model performance on with and without augmentation of test data trained only on images with traffic signals.
    2. main_completedata.ipynb: This notebook is for the visualization, model training, model performance on with and without augmentation of test data trained complete dataset.

# Dataset
 
As part of the Safe AI tech challenge, redefined the labels in the A2D2 dataset where traffic light as 1 and the rest as background 0.Out of 5000 samples only 1307 samples has traffic lights and the rest without traffic lights. Created a separate folder which contains only traffic lights labels and a folder with all the redefined lables to train two models and to evaluate the model perfomace.

The folder data_utils cotains the python scripts to create new labels (mask_generator.py) and to preprocess the data (dataloader.py, utils.py).

# Model 

For semantic segmentation the Deeplabv3Plus mobilenet model with the the pretrained weights are used which can be accesed from the git repository:  https://github.com/VainF/DeepLabV3Plus-Pytorch.git.
Fine-tuned the classifier head by modifying last layer as per the number of classes in my case its 1 and also added a dropout layer in the model.
As the dataset is highly imbalanced,trained the model on complete dataset i.e, images with and wihtout traffic lights ('new_model_weights_dropout_completedata.pth') and trained only traffic lights images('new_model_weights_dropout_04.pth).

# Inference

The iou (intersection of union) score is used to evaluate the model perfomace and Shanon's entropy to estimate the predicitive (model) uncertainty.

Inference is carried out in two approches:
1. Standard approch 
2. Dropout during inference time (Monte-Carlo dropout)
    in this approch the dropout is enabled during test time and perfomred multiple forward passes, the final output is the average of all the predictions.


# observation:
     The overall iou on the model trained with the complete dataset : 

1. main.ipynb:
2. main_complete_data.ipynb
            this 
3. train.py
4. test.py
5. metrics.py
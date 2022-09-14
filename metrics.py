import torch
import numpy as np

"""
the metric entropy is adopted from the repository : https://github.com/valeoai/ADVENT.git (ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation)
"""

def iou_metric(predicted, labels):
    """ compute intersection of union given predectied values and labels.

        :param predicted: Sigmoid activation output.
        :param labels: corresponding labels.
         
        :return : iou score
    """
    assert labels.shape == predicted.shape
    assert len(labels.shape) == 4

    predicted_threshold = predicted>0.5 #thresholding on prediction can be changed 
    intersection =np.logical_and(labels, predicted_threshold)
    union = np.logical_or(labels, predicted_threshold)
    eps = 1e-10
    iou = np.sum(intersection + eps)/ np.sum(union + eps)

    return iou

def entropy(prob):
    """given probability values computes shanon's entropy
    """
    
    n, c, h, w = prob.shape

    entropy_ = -np.sum(np.multiply(prob, np.log2(prob + 1e-8)), axis= 1) / (n * h * w * np.log2(c + 1e-10))
   # entropy_ = -torch.sum(torch.mul(prob, torch.log2(prob+ 1e-30)), dim = 1) / (n * h * w * np.log2(c + 1e-10))

    return entropy_ 
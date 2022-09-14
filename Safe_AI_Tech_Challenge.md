# Safe AI Tech Challenge

Autonomous vehicles will likely play an essential role in transforming future urban mobility. In this mini project, you will explore a common task in automated driving: using semantic segmentation to classify objects based on camera images. As safety is a critical problem in this context, you are also asked to quantify the uncertainty in your model’s prediction. Additionally, within a short presentation and an ensuing discussion, we would love to hear your views regarding the sensor inputs that can be used to obtain a good environment model surrounding the vehicle.


## Rules and requirements

* Send us your solution at least 48h hours before your tech interview.
* Do not spend more than a maximum of 6h to solve the tasks.
* Be prepared for a code review and questions about your code and the algorithms used.
* Document all necessary steps to run your solution and follow good coding practices.
* Complete the tasks using Python and either PyTorch or TensorFlow.
* Proprietary tools which are not publicly available are excluded, but you are free to use open-source tools and frameworks.
* You do not need. to use heavy compute resources for this task For example, you can run your code for a few iterations on a CPU to test it.


## Task 1

1. Download the dataset from https://web.tresorit.com/l/RrETU#Csw625bSkIWAMyuHKlMBSA and familiarize yourself with it. The provided dataset is a compressed version of the A2D2 dataset and purely for experimental purposes. You can find documentation for the original A2D2 dataset at https://www.a2d2.audi.
2. Perform an analysis of the dataset including, e.g., class distribution and number of instances per class.
3. For the rest of this task, we are only interested in the segmentation of traffic lights and hence we have a binary segmentation problem. Redefine the label masks such that traffic light labels are 1 and other classes/background are 0.
4. Visualize a few images containing traffic lights together with their binary masks.
5. Choose an appropriate split of the dataset with a 70:15:15 train:val:test ratio.
6. Pick a fast semantic segmentation model of your choice that was trained on the Cityscapes dataset. Document the source of both the model and the weights.
7. Take the necessary steps to use the pre-trained model for the binary segmentation problem at hand.
8. Fine-tune the pre-trained model on the provided dataset. You don't need to run a full training - just make sure that your training pipeline works correctly.
9. Use an appropriate set of metrics to evaluate your model's performance.
10. Implement a safety-related uncertainty metric and evaluate it on your test set. Be prepared to discuss an additional metric.

### Discussion Points

Among other things, we might discuss the following points during the interview:
* Justify your choice of loss function and selection of evaluation metrics.
* How would you verify the safety of the developed function?


## Task 2

Prepare a 5-minute presentation concerning the safety relevance of using multiple sensors (e.g., camera, radar, and lidar) for ADAS/AD functions. This will be followed by a discussion of the topic.

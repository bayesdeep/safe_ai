import os,sys
from PIL import Image
import torch.utils.data as data
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class A2d2dataset(data.Dataset):
    """
    A2d2dataset is a class which loads and preprocess the data

    """ 
    def __init__(self, root:str , image_folder:str, labels_folder:str, 
                    transforms = None):
        """
        :param root (string): path to the root directory of the dataset.
        :param image_folder (string): Name of the  folder which contains images in the root directory.
        :param labels_folder (String): Name of the folder which contains labels in the root directory.
        :param transforms (optional [callable]): A function that performs transformation on the given input.
                        For instance transforms.ToTensor:
                                     transforms.Resize:

        :param labels_trasforms (optional [callable]): A function that performs transformation on the lables.
        """
        self.root = root
        self.image_folder = image_folder
        self.labels_folder = labels_folder
        self.transforms = transforms
        # self.labels_transform = labels_transform
        self.images = []
        self.labels =[]

        root_directory = os.path.join(self.root)
        image_directory= os.path.join(root_directory, self.image_folder)
        labels_directory= os.path.join(root_directory, self.labels_folder)

        for images, labels in zip(sorted(os.listdir(image_directory)), sorted(os.listdir(labels_directory))):
            self.images.append(os.path.join(image_directory, images))
            self.labels.append(os.path.join(labels_directory, labels))

            
    def __len__(self):
        """
        :returns: number of samples in the directory 
        """
        return  len(self.images)

    def __getitem__(self, index: int):

        images = Image.open(self.images[index]).convert("RGB")     # to apply some transformations on the images it is necessary to be in either ndarray or PIL image format
        labels = read_image(self.labels[index], mode=ImageReadMode.GRAY) # readimg labels directly usiny read_image function from pytorch assuming we won't be applying any transformations
       
        if self.transforms:
            images = self.transforms(images)

        return  images, labels

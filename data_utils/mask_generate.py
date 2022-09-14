import os
import json
from PIL import Image, ImageColor
import numpy as np

"""mask_generator.py, using this script we can create a new labels for the target classes
"""

def json_reader(path = 'a2d2_compressed/class_list.json', target_class= 'Traffic signal'):

    """Read json file.
        :return: Dictionary with the specific class
    """
    file = open(path)
    load_file = json.load(file)
    new_class_list = {}
    for colours, classes in load_file.items():
        if target_class in classes:
            new_class_list.update({colours: 1})
    return new_class_list


def obtain_new_masks(labels):
    """ creates a new masks.
    """
    mask = np.zeros((labels.shape[0], labels.shape[1]), dtype= np.uint8)
    new_class_list = json_reader()
    for keys , values in new_class_list.items():
        current_color = ImageColor.getrgb(keys)
        create_binary_mask = np.all(labels == current_color, axis = -1)
        mask[create_binary_mask] = values
    return mask


def binary_mask(root_directory, image_directory, labels_directory):  

    """read through all the samples and sort and save samples which contains target class in anew folder
    """
    for image,  label in zip(sorted(os.listdir(image_directory)), sorted(os.listdir(labels_directory))):
        images =  np.asarray(Image.open(os.path.join(image_directory, image)))
        labels = np.asarray(Image.open(os.path.join(labels_directory, label)))
        new_labels = obtain_new_masks(labels)
        if np.all(np.unique(new_labels) == np.array([0,1])):
            print(np.unique(new_labels))
            img2png = Image.fromarray(images, mode = 'RGB')
            img2png.save(os.path.join(root_directory +'/'+ 'images_traffic', image))
            to_png = Image.fromarray(new_labels, mode = 'L')
            to_png.save(os.path.join(root_directory +'/'+ 'binary_labels_traffic', label))


if __name__ == "__main__":

    root_directory = os.path.join('a2d2_compressed')
    image_directory= os.path.join(root_directory, 'images')
    labels_directory= os.path.join(root_directory,'labels')

    
    if not os.path.exists(root_directory +'/'+ 'binary_labels_traffic'):
        os.makedirs(root_directory +'/'+ 'binary_labels_traffic')
    if not os.path.exists(root_directory +'/'+ 'images_traffic'):
        os.makedirs(root_directory +'/'+ 'images_traffic')

    binary_mask(image_directory, labels_directory)
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def train_val_test(dataset,test_split = 0.15):
    
    """split the dataset into train valid and test.

        :param dataset: dataset which we would like to split
        :param test_split (float): ratio of test data

        :return: dictionary containing the split dataset
    """
    train_idx, test_idx = train_test_split(list(range(len(dataset))),test_size= test_split, shuffle = True, random_state= 1)
    train_idx, val_idx = train_test_split(range(len(train_idx)), test_size = 0.15,shuffle= True, random_state= 1)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

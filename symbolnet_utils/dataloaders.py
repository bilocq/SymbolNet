"""
EMNIST data loaders for training and testing SymbolNet
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from symbolnet_utils.emnist_tools import emnist, EmnistClassLabelsSetup, indices_finder



class EmnistPairsLoader():
    """
    Main "data loader" used for training. Calling an instance of this class will return two batches of EMNIST images
    (as tensors batch_1 and batch_2) along with a one-dimensional "target" tensor containing 0's and 1's. 
        target[i]==0 means batch_1[i] and batch_2[i] are from different classes.
        target[i]==1 means batch_1[i] and batch_2[i] are from the same class.

    REQUIRED PARAMETERS:
        - data_path: (string) Path to EMNIST dataset. If dataset isn't already there, it gets downloaded.
        
    OPTIONAL PARAMETERS: 
        - device: Can be either torch.device('cpu') or torch.device('cuda').
                  TYPE: torch.device  
                  DEFAULT: torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        - train: (bool) True means we use EMNIST training data, False means we use EMNIST test data. 
                 DEFAULT: True
        - split: (string) Which split of the EMNIST dataset to use.
                 DEFAULT: 'byclass'
        - classes_string: (string) A string specifying the classes to be used. See symbolnet_utils.emnist_tools.EmnistClassLabelsSetup
                          for a detailed explanationof how to specify other sets of classes.
                          DEFAULT: 'digits'
        - batch_size: (int) Number of elements per batch, where each element contains a pair of images along with a 
                      label which is either 0 (images are from different classes) or 1 (images are from same class).
                      DEFAULT: 64
        - inds_save_folder: (string) The path to a folder where a file containing the list of indices for each class
                            is either already present or will be saved. Finding the dataset indices of all the elements
                            of the requested classes requires looping through the dataset, which can take some time.
                            Saving the resulting lists of indices allows us to avoid doing this each time an instance
                            of this class is created.
                            DEFAULT: None (meaning no such file is saved or loaded).

    A dataloader object is defined for each class specified by "class_labels", and is used to iterate over the samples
    from that class.
    
    The "p" argument of __call__ is a float between 0 and 1 (inclusively). It decides the probability that any given pair 
    of samples is from the same class. p==0 means all pairs are from different classes, while p==1 means all pairs are 
    from the same class.
    """
    
    def __init__(self, data_path, **kwargs):
        # Setup
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        train               = kwargs['train']            if ('train'            in kwargs) else True
        split               = kwargs['split']            if ('split'            in kwargs) else 'byclass'
        classes_string      = kwargs['classes_string']   if ('classes_string'   in kwargs) else 'digits'
        inds_save_folder    = kwargs['inds_save_folder'] if ('inds_save_folder' in kwargs) else None
        self.batch_size     = kwargs['batch_size']       if ('batch_size'       in kwargs) else 64
    
        class_labels, classes = EmnistClassLabelsSetup(split, classes_string, verbose=False)
        self.classes = classes
        self.class_labels = class_labels
        
        ### PER CLASS DATALOADERS 
        # First we find all indices of samples from each class specified in class_labels.
        train_or_test ='train' if train else 'test'
        split_key = f'{split}_{train_or_test}'
        full_indices_dict = indices_finder(data_path, split, classes_string, train, save_folder=inds_save_folder)
        indices_dict = full_indices_dict[split_key]
        
        # Now that we have the list of indices for each class, we make a sub-dataset and a corresponding dataloader
        # for each class. We record the dataloaders in a dict where the keys are the class labels.
        ds = emnist(data_path, split, train=train)
        self.class_loaders = {}
        for i in range(len(classes)):
            c_dataset = Subset(ds, indices_dict[classes[i]])
            c_loader = DataLoader(c_dataset, 
                                  batch_size = None,
                                  shuffle = True,
                                  pin_memory = (self.device.type=='cuda'))
            self.class_loaders[class_labels[i]] = [c_loader, iter(c_loader)]  # We will re-iterate over the first element when the second is exhausted.
        
        
    def __call__(self, p=0.5):
        target = torch.bernoulli(torch.ones(self.batch_size)*(1-p)).type(torch.LongTensor)
        batch_1 = [] 
        batch_2 = []
        for i in range(self.batch_size):
            if target[i]==0: # target[i]==0 means batch_1[i] and batch_2[i] are from different classes
                cs = list(np.random.choice(self.class_labels, size=2, replace=False)) 
            else: # target[i]==1 means batch_1[i] and batch_2[i] are from the same class.
                cs = list(np.random.choice(self.class_labels, size=1))*2
            try:
                sample_1 = next(self.class_loaders[cs[0]][1])
            except StopIteration: # Just ran out of samples from that class, so we re-iterate over the dataloader.
                self.class_loaders[cs[0]][1] = iter(self.class_loaders[cs[0]][0])
                sample_1 = next(self.class_loaders[cs[0]][1])
            try:
                sample_2 = next(self.class_loaders[cs[1]][1])
            except StopIteration: # Just ran out of samples from that class, so we re-iterate over the dataloader.
                self.class_loaders[cs[1]][1] = iter(self.class_loaders[cs[1]][0])
                sample_2 = next(self.class_loaders[cs[1]][1])
            batch_1.append(sample_1[0])
            batch_2.append(sample_2[0])
        
        batch_1 = torch.stack(batch_1).to(self.device)
        batch_2 = torch.stack(batch_2).to(self.device)
        return batch_1, batch_2, target.to(self.device)



def EmnistSinglesLoader(data_path, **kwargs):
    """
    Returns a standard torch.utils.data.Dataloader object for the given EMNIST split and classes (specified through 
    classes string). Useful for classification tests.
    
    REQUIRED PARAMETERS
        - data_path: (string) Path to EMNIST dataset. If dataset isn't already there, it gets downloaded.
        
    OPTIONAL PARAMETERS
        - train: (bool) True means we use EMNIST training data, False means we use EMNIST testing data.
                 DEFAULT: True
        - split: (string) Which split of the EMNIST dataset to use.
        - classes_string: (string) A string specifying the classes to be used. See symbolnet_utils.emnist_tools.EmnistClassLabelsSetup
                          for a detailed explanationof how to specify other sets of classes.
        - num_images: (int) Total number of images to load.
                      DEFAULT: -1, meaning the full dataset split is loaded.
        - batch_size: (int or None) Nummber of images per batch.
                      DEFAULT: None. This means elements are not batched, it is more convenient for classification tests.
        - inds_save_folder: (string) The path to a folder where a file containing the list of indices for each class
                            is either already present or will be saved. Finding the dataset indices of all the elements
                            of the requested classes requires looping through the dataset, which can take some time.
                            Saving the resulting lists of indices allows us to avoid doing this each time an instance
                            of this class is created.
                            DEFAULT: None (meaning no such file is saved or loaded).
    """
    if 'device' in kwargs.keys():
        device = kwargs['device']
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train            = kwargs['train']            if ('train'            in kwargs) else False
    split            = kwargs['split']            if ('split'            in kwargs) else 'byclass'
    classes_string   = kwargs['classes_string']   if ('classes_string'   in kwargs) else 'digits'
    num_images       = kwargs['num_images']       if ('num_images'       in kwargs) else -1 # -1 means we'll go over all data.
    batch_size       = kwargs['batch_size']       if ('batch_size'       in kwargs) else None # None means loader returns individual images.
    inds_save_folder = kwargs['inds_save_folder'] if ('inds_save_folder' in kwargs) else None
    class_labels, classes = EmnistClassLabelsSetup(split, classes_string, verbose=False)

    # Find all indices of samples that belong to one of the specified classes.
    train_or_test ='train' if train else 'test'
    split_key = f'{split}_{train_or_test}'
    full_indices_dict = indices_finder(data_path, split, classes_string, train, save_folder=inds_save_folder)
    indices_dict = full_indices_dict[split_key]
    inds = []
    for c in classes:
        inds += indices_dict[c]
    random.shuffle(inds)
    
    # Shorten inds if necessary
    if 0 < num_images < len(inds):
        inds = inds[:num_images]
    if num_images > len(inds):
        print(f'The requested number of images is larger than the test images available for given split and classes.\nNumber of images decreased to {len(inds)}.')
    
    # Sub-dataset corresponding to the indices obtained, and a dataloader for it.
    ds = emnist(data_path, split, train=train)
    inds_dataset = Subset(ds, inds)
    loader = DataLoader(inds_dataset, 
                        batch_size = batch_size,
                        shuffle = False,
                        pin_memory = (device.type=='cuda'))
    loader.classes = classes
    loader.class_labels = class_labels
    loader.num_images = len(inds)
    return loader
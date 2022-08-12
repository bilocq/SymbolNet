"""
Some EMNIST-specific tools
"""

import os
from tqdm import tqdm
import pickle
import random
import string
import torch
from torchvision import datasets, transforms



class TransposeTensor():
    """
    The images from the EMNIST dataset downloaded using torchvision.datasets.EMNIST come flipped.
    We unflip them with this.
    """
    def __call__(self, sample):
        return torch.transpose(sample,1,2)



def emnist(data_path, split, train=True):
    """
    We load the EMNIST dataset with the built-in torchvision dataset class
    
    ARGUMENTS
        - root: (string) Path to the EMNIST dataset. If the dataset isn't already there, it gets downloaded.
        - split: (string) Which split of the EMNIST dataset to use
        - train: (bool) True means training data, False means testing data.
    
    OUTPUT
        An instance of the torchvision.datasets.EMNIST class
    """
    transpose_image = transforms.Compose([transforms.ToTensor(), TransposeTensor()]) 
    ds = datasets.EMNIST(data_path,
                         split=split,
                         transform=transpose_image,
                         train=train,
                         download=True)
    return ds



def EmnistClassLabelsSetup(split, classes_string, verbose=False):
    """
    Finds and returns the integer class labels corresponding to the classes specified by 'classes_string' within the given
    'split' of the EMNIST dataset.
    
    ARGUMENTS:
        - split:  (string) Which split of the EMNIST dataset to use. The 'letters' split is not supported because it merges
                  corresponding uppercase and lowercase letters.
        - classes_string: (string) A string specifying the classes for which the corresponding integer labels must be returned.
                          This  string must be composed of one or several groups of classes separated by underscores (_). Each
                          group can have one of the following forms:
                            - One of the default groups: 'digits', 'uppercases' and 'lowercases'. Then all classes of this
                              type will be included
                            - Limits separated by a dash (-), eg '0-20', 'A-H' or 'b-t'. 
                                . If the limits are integers, all integers between and including the limits are included.
                                . If the limits are uppercase or lowercase letters, then all letters alphabetically between
                                  the limits and including the limits within the split are included. The limits should be
                                  either both uppercase letters or both lowercase letters, in alphabetical order. Note that 
                                  the 'bymerge' and 'balanced' splits of EMNIST do not include a class for each lowercase 
                                  letter, so when using one of these splits lowercase classes options are reduced.
                            - A sequence of characters each specifying one class. Each character should be either a digit, an
                              uppercase letter or a lowercase letter included in the split.
                            - 'all'. All classes of the split are included
                          EXAMPLES: classes_string = 
                            'digits': Only digit classes (i.e. 0 to 9) are included
                            '0-5_lowercases': Digits 0 to 5 (inclusively) are included, along with all lowercase letter classes.
                            'A-D_d-f_024Z': Classes 'A', 'B', 'C', 'D', 'd', 'e', 'f', '0', '2', '4', 'Z' are included.
        - verbose: (bool) When True, all included class labels and corresponding classes will be printed.
    
    OUTPUT
        A sorted list of class labels with any duplicate label removed, along with a corresponding of class names.
        Class labels are integers, while class names are strings. For example, the class with label 3 has name '3', while the
        class with label 11 has name 'B' (in any split except 'digits').
    """
    # Sanity check for split and class_labels arguments
    assert split != 'letters', "The emnist split 'letters' is not supported"
    assert split in ['byclass', 'bymerge', 'balanced', 'digits'], 'dataloader given invalid EMNIST split.'
    
    # Limiting indices for each split
    d = {'byclass':  [0,10,36,62], # 0-9: digits, 10-35: uppercases, 36-61: lowercases
         'bymerge':  [0,10,36,47],
         'balanced': [0,10,36,47],
         'digits':   [0,10,10,10]} # No letters in this split.
    
    uppercases = string.ascii_uppercase
    lowercases = string.ascii_lowercase
    lowercases_ = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'] # The 'balanced' and 'bymerge' splits only have these lowercase classes
    
    class_labels = []
    default_groups = ['digits', 'uppercases', 'lowercases']
    
    for group in classes_string.split('_'):
       
        ## Default groups
        if group in default_groups:
            low_ind = d[split][default_groups.index(group)]
            up_ind = d[split][default_groups.index(group)+1]
            group_labels = list(range(low_ind, up_ind))
            if not group_labels: # group_labels is empty
                raise ValueError(f"Class group {group} is invalid for '{split}' split.")
            class_labels.extend(group_labels)
        
        ## Groups specified by limits
        elif '-' in group:
            lims = group.split('-')
            assert len(lims)==2
            # Integers: these can also be used to specify the labels of uppercase and lowercase classes in the chosen split.
            if all(lim.isdigit() for lim in lims):
                lims = [int(lim) for lim in lims]
                assert (lims[0] >= 0) and (lims[0] < lims[1] < d[split][-1]), f"The given class group {group} is invalid for the '{split}' split."
                group_labels = list(range(lims[0], lims[1]+1))
                class_labels.extend(group_labels)
            # Uppercases
            elif all(lim in uppercases for lim in lims):
                lims = [uppercases.index(lim)+d[split][1] for lim in lims]  
                assert lims[0] < lims[1], f'{group} is an invalid group of uppercases. Group limits should be in alphabetical order.'
                group_labels = list(range(lims[0], lims[1]+1))
                class_labels.extend(group_labels)
            # Lowercases for byclass split
            elif split=='byclass' and all(lim in lowercases for lim in lims):
                lims = [lowercases.index(lim)+d[split][2] for lim in lims]
                assert lims[0] < lims[1], f'{group} is an invalid group of lowercases. Group limits should be in alphabetical order.'
                group_labels = list(range(lims[0], lims[1]+1))
                class_labels.extend(group_labels)
            # Lowercases for bymerge and balanced splits
            elif (split in ['bymerge', 'balanced']) and all(lim in lowercases_ for lim in lims):
                lims = [lowercases_.index(lim)+d[split][2] for lim in lims]
                assert lims[0] < lims[1], f'{group} is an invalid group of lowercases. Group limits should be in alphabetical order.'
                group_labels = list(range(lims[0], lims[1]+1))
                class_labels.extend(group_labels) 
            # Raise error if nothing worked
            else:
                raise ValueError(f"Class group {group} is invalid for '{split}' split.")

        ## Groups specified by listing all classes
        else:
            for c in list(group):
                if c.isdigit():
                    class_labels.append(int(c))
                elif c in uppercases:
                    class_labels.append(uppercases.index(c)+d[split][1])
                elif split=='byclass' and (c in lowercases):
                    class_labels.append(lowercases.index(c)+d[split][2])
                elif (split in ['bymerge', 'balanced']) and (c in lowercases_):
                    class_labels.append(lowercases_.index(c)+d[split][2])
                else:
                    raise ValueError(f"Class group {group} is invalid for '{split}' split.")
    
    # Remove duplicates and sort
    class_labels = sorted(list(dict.fromkeys(class_labels)))  
    classes = []
    for cl in class_labels:
        if cl < d[split][1]:
            classes.append(str(cl))
        elif cl < d[split][2]:
            classes.append(uppercases[cl - d[split][1]])
        else:
            if split=='byclass':
                classes.append(lowercases[cl - d[split][2]])
            else:
                classes.append(lowercases_[cl - d[split][2]])
    
    if verbose:
        print(f'Class labels: \n{class_labels}')
        print(f'These correspond to the classes: \n{classes}')
   
    return class_labels, classes



def indices_finder(data_path, split, classes_string, train, save_folder=None):
    """
    Loads the requested 'split' of the EMNIST dataset, finds the dataset indices of the elements from classes
    specified in 'classes_string', and records these indices in a dictionary. This is useful notably for the 
    data loaders in symbolnet_utils.dataloaders. It can take some time to go through the entire dataset, so the
    resulting dictionary can be save under 'indices.pkl' in 'save_folder', if this argument is specified. 
    Accordingly, if 'save_folder' is specified and contains an 'indices.pkl' file, loads the dictionary from
    this file and checks if it contains the indices from the correct split and classes before going through 
    the dataset.
    
    REQUIRED PARAMETERS:
        - data_path: (string) Path to EMNIST dataset. If dataset isn't already there, it gets downloaded.
        - split: (string) Which split of the EMNIST dataset to use.
        - classes_string: (string) A string specifying the classes to be used. See above under EmnistClassLabelsSetup
                          for a detailed explanationof how to specify other sets of classes.
        - train: (bool) True means training data, False means we testing data.
    
    OPTIONAL PARAMETER:
        - save_folder: (string) Path to a folder where to save the indices dictionary, under 'indices.pkl', if that 
          file doens't already exist. If that file does already exist, the indices dictionary it contains is loaded
          and the function checks whether it contains the indices from the correct split and all classes. If it does,
          no need to go through the dataset again. If it doesn't, the function goes through the dataset and appends
          the indices for the missing classes to the dictionary before saving it again in the 'indices.pkl' file.
    """    
    # Setup loader
    class_labels, classes = EmnistClassLabelsSetup(split, classes_string)
    ds = emnist(data_path, split, train=train)
    indices_loader = torch.utils.data.DataLoader(ds,                  # We loop over the data using "indices_loader" 
                                                 batch_size = 1000,   # because this is faster than looping over the
                                                 drop_last = False,   # dataset itself.
                                                 num_workers = 0)
    
    # Prepare dictionary to save indices
    train_or_test ='train' if train else 'test'
    split_key = f'{split}_{train_or_test}'
    indices_file = os.path.join(save_folder, 'indices.pkl') if (save_folder is not None) else None
    if (indices_file is not None) and os.path.exists(indices_file):
        with open(indices_file, 'rb') as file:
            full_indices_dict = pickle.load(file) 
        if split_key in full_indices_dict.keys():
            split_indices_dict = full_indices_dict[split_key]
        else:
            split_indices_dict = {} 
    else:
        full_indices_dict = {}
        split_indices_dict = {}

    # Remove classes that are already in indices_dict from 'class_labels' and 'classes'(relevant when resume==True).
    to_remove = []
    for i in range(len(classes)):
        if not classes[i] in split_indices_dict.keys():
            split_indices_dict[classes[i]] = []
        else:
            to_remove.append(i)
    to_remove.reverse()
    for ind in to_remove:
        del class_labels[ind]
        del classes[ind]
    
    # Stop if there are no classes left (i.e. all required classes were already in the loaded file).
    if not classes: # 'classes' is empty
        return full_indices_dict
    
    # Main loop
    print("Finding all indices for given classes in dataset...\n")
    for batch_ind, (_, labels) in enumerate(tqdm(indices_loader)):
        for i in range(len(classes)):
            new_inds = (labels == class_labels[i]).nonzero(as_tuple=True)[0] + batch_ind * 1000
            split_indices_dict[classes[i]] += new_inds.tolist()
    
    # Record and save
    full_indices_dict[split_key] = split_indices_dict
    if (indices_file is not None):
        with open(indices_file, 'wb') as file:
            pickle.dump(full_indices_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return full_indices_dict


    
def examples_assembly(ds, class_labels, num_examples=10, shuffled=True, device=None):
    """
    Returns a list of tensors of length len(class_labels). The ith element of the list is a batch containing
    num_examples images taken from the dataset ds and labeled class_labels[i].
    
    This function is useful when classifying images with SymbolNet. A test image can be compared to 
    each image from each batch, and can be classified according to the class label that yielded the highest
    average similarity.
    
    REQUIRED PARAMETERS
        - ds: A dataset. Each element ds[i] should be a tuple or list of at least two elements, the first 
              being a (tensorized) image and the second being a (integer) label.
        - class_labels: A list of integers specifying the labels of the requested classes.
    
    OPTIONAL PARAMETERS
        - num_examples: Number of example images from each specified class.
                         TYPE: int
                         DEFAULT: 10
        - shuffled: Whether to shuffle the dataset before looking from specimens.
                    TYPE: bool
                    DEFAULT: True
        - device: Can be either torch.device('cpu') or torch.device('cuda').
                  TYPE: torch.device  
                  DEFAULT: torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    examples = [[] for _ in range(len(class_labels))]
    inds = list(range(len(ds)))
    if shuffled:
        random.shuffle(inds)
    
    i = 0
    while any(len(l) < num_examples for l in examples):
        ind = inds[i]
        label = ds[ind][1]
        if (label in class_labels) and (len(examples[class_labels.index(label)]) < num_examples):
            examples[class_labels.index(label)].append(ds[ind][0])
        i += 1
    
    for j in range(len(class_labels)):
        examples[j] = torch.stack(examples[j]).to(device)
        
    return examples



def sort_classes(list_of_classes):
    """
    Returns a sorted version of list_of_classes, where digits come first, followed by uppercase letters then 
    lowercase letters and finally classes that do not fit in any of these categories (these extra classes do
    not get sorted, i.e. they appear in the sorted list in the same order as in the input list).                                                                                  
    
    PARAMETERS:
        - list_of_classes: A list of strings
    """
    all_chars = [str(d) for d in range(10)] + list(string.ascii_uppercase) + list(string.ascii_lowercase)
    sorted_list  = [c for c in all_chars if c in list_of_classes]     # Digits, uppercases, lowercases
    sorted_list += [c for c in list_of_classes if c not in all_chars] # Other classes
    sorted_list = list(dict.fromkeys(sorted_list))                    # Removes duplicates (for the other classes)
    return sorted_list
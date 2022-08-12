"""
Tools for saving feature representations of classes with SymbolNet
"""

import os
import pickle
import torch



def features_to_cpu(features):
    """
    Creates a copy of 'features' where the tensors are in CPU memory
    
    PARAMETERS:
        - features: (dict) The output of a feature network (from networks.feature_network) when the input is a batch
                    of images from class_.
    """
    features_ = {}
    for key in features.keys():
        features_[key] = features[key].cpu()
    return features_



def register_features(folder, class_, features):
    """
    Creates the 'memory.pkl' file in 'folder', if it doesn't already exist; otherwise, loads the memory dictionary
    it contains. Adds 'features' to the dictionary under 'class_'. If the class_ key already exists, this means the 
    tensors contained in features are appended to the tensors contained in dict[class_] alog the batch dimension.
    
    PARAMETERS:
        - folder: (string) Folder where to save (and load, if it exists already) the 'memory.pkl' file.
        - class_: (string) The name of the class for the features to memorize.
        - features: (dict) The output of a feature network (from networks.feature_network) when the input is a batch
                    of images from class_.
    """
    # Put features on cpu
    features = features_to_cpu(features)
    # Load memory 
    memory_file = os.path.join(folder, 'memory.pkl')
    if 'memory.pkl' in os.listdir(folder):
        with open(memory_file, 'rb') as f:
            memory_dict = pickle.load(f)
    else:
        memory_dict = {}   
    # Add new features to memory
    if class_ in memory_dict.keys(): # If class_ already present in memory, append new features to memory tensors along batch dimension
            class_dict = memory_dict[class_]
            for key in class_dict.keys():
                class_dict[key] = torch.cat([class_dict[key], features[key]], dim=0)
            memory_dict[class_] = class_dict
    else:
        memory_dict[class_] = features
    # Save memory
    with open(memory_file, 'wb') as f:
        pickle.dump(memory_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
      
    

def register_images(folder, network, imgs, classes):
    """
    Behaves similarly to register_features, but obtains features by mapping the images in 'imgs" through the feature
    network of 'network'. Also, 'imgs' can be a list of batches of images and 'classes' a corresponding list of class
    names.
    
    PARAMETERS:
        - folder: (string) Folder where to save (and load, if it exists already) the 'memory.pkl' file.
        - network: An instance of the networks.SymbolNet class.
        - imgs: (torch.tensor or list of torch.tensors) A batch of images from the same class, or a list of batches of
                images with each batch containing images from a given class.
        - classes: (string or list of strings) A class name or a list of class names corresponding to the imgs input
                   (so classes[i] should be the class name for imgs[i]).
    """
    # Sanity checks 1
    if torch.is_tensor(imgs):
        imgs = [imgs]
        assert type(classes) == str, "If input 'imgs' is a tensor, 'classes' should be a string."
        classes = [classes]
    assert len(imgs) == len(classes), "If inputs 'imgs' and 'classes are lists or tuples, they should be the same lenght."
    
    # Load memory 
    memory_file = os.path.join(folder, 'memory.pkl')
    if 'memory.pkl' in os.listdir(folder):
        with open(memory_file) as f:
            memory_dict = pickle.load(f)
    else:
        memory_dict = {}  
    
    for i in range(len(imgs)):
        img = imgs[i]
        class_ = classes[i]

        # Sanity checks 2
        error_message = f"Input image should be a tensor of size (1,28,28) or (n,1,28,28), got {tuple(img.shape)}."
        assert torch.is_tensor(img), f"Input image should be a tensor, got a {type(img)}."
        assert type(class_) == str, f"Input 'class_' should be a string, got a {type(class_)}."
        assert img.dim() in [3,4], error_message
        assert list(img.shape[-2:]) == [28,28], error_message
        assert img.shape[-3] == 1, error_message

        # Resize if necessary
        if img.dim() == 3:
            img = img.view((1,1,28,28))

        # Get features and save them in memory    
        features = features_to_cpu(network.feature_network(img))
        if class_ in memory_dict.keys(): # If class_ already present in memory, append new features to memory tensors along batch dimension
            class_dict = memory_dict[class_]
            for key in class_dict.keys():
                class_dict[key] = torch.cat([class_dict[key], features[key]], dim=0)
            memory_dict[class_] = class_dict
        else:
            memory_dict[class_] = features
        
    # Save memory
    torch.save(memory_dict, memory_file)    



def delete_class(folder, class_):
    """
    Deletes class_ from memory.pkl
    """
    # Load memory 
    memory_file = os.path.join(folder, 'memory.pkl')
    assert os.path.isfile(memory_file), "No memory.pkl file found in given folder"
    with open(memory_file) as f:
        memory_dict = pickle.load(f)
    # Delete class
    memory_dict.pop(class_, None)
    # Save memory 
    torch.save(memory_dict, memory_file)
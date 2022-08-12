import os
import argparse
import logging
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch

from symbolnet_utils import dataloaders, networks, memory, folder_tool, emnist_tools




def register_memory(folder, data_path, classes, **kwargs):
    """
    Creates the file 'memory.pkl' inside 'folder' and saves the feature representations of images from the classes
    specified by 'classes' in it. This memory file contains a pickled dictionary (let's call it memory_dict) whose keys
    are class names. For instance, if the class 'A' is specified in 'classes', then memory_dict['A'] contains the output
    features obtained when mapping 'num_mem' randomly selected images labeled 'A' through the network's feature_network
    (except when using K-means, i.e. when 'num_see' > 'num_mem', see below). These feature representations can then be 
    compared, using the network's evaluator, to the features of images to be classified (see e.g. classification_test). 
    
    Note that it is possible to memorize classes that were not used during training!
    
    REQUIRED PARAMETERS
        - folder: A folder containing the network state dictionary for which to register memory, as well as the 
                  architecture file for that state dictionary. (Both the required files 'network_state_dict.pth'
                  and 'network_info.pkl' are created when running 'train.py'.)
        - data_path: Path where the EMNIST dataset is located. The data_root path must lead to a folder containing
                     a folder called EMNIST which itself contains the dataset. If the dataset is not there, then
                     it gets downloaded automatically.
        - classes: A string specifying which classes to use for training. Basic options are 'digits', 'uppercases'
                   and 'lowercases'. See symbolnet_utils.emnist_tools.EmnistClassLabelsSetup for a detailed explanation
                   of how to specify other sets of classes.
                   TYPE: string
    
    OPTIONAL PARAMETERS
        - device: Can be either torch.device('cpu') or torch.device('cuda').
                  TYPE: torch.device  
                  DEFAULT: torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        - emnist_split: Which split of the EMNIST dataset to use. Can be either 'byclass', 'bymerge', 'balanced'  
                        or 'digits' (the 'letters' split is not supported).
                        TYPE: string
                        DEFAULT: 'byclass'
        - train: True means we use EMNIST training data, False means we use EMNIST test data.
                 TYPE: bool
                 DEFAULT: True
        - num_mem: Number of sets features to memorize for each class. See K-MEANS explanation below
                   TYPE: int
                   DEFAULT: 10
        - num_see: Number of images to look at for each at for each class. See K-MEANS explanation below
                   TYPE: int
                   DEFAULT: 10
    
    K-MEANS OPTION
        When 'num_mem' == 'num_see' (the default option), each set of features being memorized is obtained by mapping
        one randomly selected image from the dataset through the newtwork's feature_network module. When 'num_mem' <
        'num_see', 'num_see' images are mapped through the feature network for each class, and a K-means algorithm with
        'num_mem' centroids is performed on the resulting features. The 'num_mem' centroid features are then saved.
    """
    # -------------------------- SETTING UP --------------------------
    # Folder
    required_files = ['network_state_dict.pth', 'network_info.pkl']
    resume = folder_tool.check_folder(folder, required_files)
    if not resume:
        raise ValueError('...')
    
    # Device
    if 'device' in kwargs.keys():
        device = kwargs['device']
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Dataloader - batch_size=None by default, so we load one element at a time.
    train = kwargs['train']        if ('train'        in kwargs) else True
    split = kwargs['emnist_split'] if ('emnist_split' in kwargs) else 'byclass'
    loader = dataloaders.EmnistSinglesLoader(data_path = data_path,
                                             split = split,
                                             classes_string = classes,
                                             inds_save_folder = folder,
                                             train=train,
                                             device = device) 
    loader_it = iter(loader)
    classes = loader.classes
    class_labels = loader.class_labels
    train_or_test = 'train' if train else 'test'
    
    # Network
    with open(os.path.join(folder, 'network_info.pkl'), 'rb') as net_dict_file:
        net_dict = pickle.load(net_dict_file)
    arch      = net_dict['architecture']
    comp_mode = net_dict['compare_mode']
    blocks    = net_dict['blocks']
    arch   = getattr(networks, arch)
    blocks = [int(b) for b in blocks.split(',')]
    net = arch(blocks=blocks, compare_mode=comp_mode).to(device)
    net.load_state_dict(torch.load(os.path.join(folder, 'network_state_dict.pth'), map_location=device))
    
    if 'memorized_classes' in net_dict.keys():
        mem_classes = net_dict['memorized_classes'] + classes
    else:
        mem_classes = classes
    mem_classes = emnist_tools.sort_classes(mem_classes)
    net_dict['memorized_classes'] = mem_classes
    
    with open(os.path.join(folder, 'network_info.pkl'), 'wb') as net_dict_file:
            pickle.dump(net_dict, net_dict_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Numbers of images
    num_mem = kwargs['num_mem'] if ('num_mem' in kwargs) else 10
    num_see = kwargs['num_see'] if ('num_see' in kwargs) else 10
    compute_kmeans = (num_see > num_mem)

    # Logging
    log_file = os.path.join(folder, 'log.txt')
    
    logging.basicConfig(level = 0)
    logger = logging.getLogger(__name__) # We use a single logger with 2 handlers.
    logger.propagate = False
    logger.setLevel(0)
    
    console_handler = logging.StreamHandler() # Writes to console
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    console_handler.setLevel(1)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file) # Writes to log.txt
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(2)
    logger.addHandler(file_handler)
    
    # Initial message
    message  = f"Registering memory of the following classes:\n{classes}\n"
    if compute_kmeans:
        message += f"For each class, looking at {num_see} images and memorizing {num_mem} elements through a K-means algorithm.\n"
        message += f"Images are taken from the '{split} {train_or_test}' EMNIST split."
    else:
        message += f"For each class, memorizing {num_mem} images taken at random from the '{split} {train_or_test}' EMNIST split."
    message +=  "\n...\n"
    logger.log(2, message)


    # -------------------------- FINDING IMAGES TO MEMORIZE --------------------------
    logger.log(1, "Finding images to memorize.\n")
    # Get examples from each class
    examples = {class_ : [] for class_ in classes}
    while any(len(mem_list) < num_see for mem_list in examples.values()):
        try:
            im, label = next(loader_it)
            class_ = classes[class_labels.index(label)]
            if len(examples[class_]) < num_see:
                examples[class_].append(im)
        except StopIteration:
            break
    for class_ in classes:
        examples[class_] = torch.stack(examples[class_], dim=0).to(device)
    
    
    # -------------------------- MEMORIZING --------------------------
    logger.log(1, "Memorizing elements from each class.\n")
    # Map examples to features, perform K-means if required, save in memory file.
    with torch.no_grad():
        for class_ in tqdm(classes):
            batch = examples[class_]
            features = net.feature_network(batch)
            if compute_kmeans:
                for feat_key in features.keys():
                    np_feats = features[feat_key].cpu().numpy()
                    shape = list(np_feats.shape)
                    np_feats = np_feats.reshape((shape[0], -1))
                    feat_kmeans = KMeans(n_clusters=num_mem).fit(np_feats).cluster_centers_
                    shape[0] = num_mem
                    feat_kmeans = feat_kmeans.reshape(shape)
                    feat_kmeans = torch.from_numpy(feat_kmeans)
                    features[feat_key] = feat_kmeans   
            memory.register_features(folder, class_, features)
    
    
    # -------------------------- EPILOGUE --------------------------
    message  = "DONE"
    message += "\n\n\n\n----------------------\n\n\n"
    logger.log(2, message)
            




if __name__ == '__main__':
    parser = argparse.ArgumentParser('...')
    # General
    parser.add_argument('--folder',      type=str,  required=True, help="Folder to create or from which to resume")
    # Dataset
    parser.add_argument('--data_path',    type=str, required=True,     help="Where the data is.")
    parser.add_argument('--classes',      type=str, required=True,     help="String specifying which classes to use.")
    parser.add_argument('--emnist_split', type=str, default='byclass', help="Which split of the EMNIST dataset to use.")
    # Memorize
    parser.add_argument('--num_mem', type=int, default=10, help='Number of elements to store in memories for each class.')
    parser.add_argument('--num_see', type=int, default=10, help='Number of images from each class to see while memorizing.')
    
    args = parser.parse_args()
    register_memory(**args.__dict__)
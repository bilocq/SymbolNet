"""
Testing a SymbolNet on whether it can be used to correctly classify images into given categories.
"""

import os
import time
import argparse
import logging
import pickle
from tqdm import tqdm
from tabulate import tabulate
import torch

from symbolnet_utils import dataloaders, networks, folder_tool, emnist_tools



def classification_test(folder, data_path, **kwargs):
    """
    Loads the state dictionary of a network from symbolnet_utils.networks and test how well it can be used to classify
    individual images into given categories. Classification is done by comparing each test image to a number of
    example images from each category using the SymbolNet, and choosing the category for which the average
    similarity was highest. In this form, this function is specifically suited to the EMNIST dataset, as it relies
    on the EmnistPairsLoader from symbolnet_utils.dataloaders.
    
    Note that SymbolNet can be used to classify images into classes that were not used during training!
    
    REQUIRED PARAMETERS
        - folder: A folder containing the state dictionary to test, under folder/network_state_dict.pth
        - data_path: Path where the EMNIST dataset is located. The data_path path must lead to a folder containing
                     a folder called EMNIST which itself contains the dataset. If the dataset is not there, then
                     it gets downloaded automatically. 
    
    OPTIONAL PARAMETERS
        - device: Can be either torch.device('cpu') or torch.device('cuda').
                  TYPE: torch.device  
                  DEFAULT: torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        - emnist_split: Which split of the EMNIST dataset to use. Can be either 'byclass', 'bymerge', 'balanced'  
                        or 'digits' (the 'letters' split is not supported).
                        TYPE: string
                        DEFAULT: 'byclass'
        - classes: A string specifying which classes to use for testing. Basic options are 'digits', 'uppercases'
                   and 'lowercases'. See symbolnet_utils.emnist_tools.EmnistClassLabelsSetup for a detailed explanation
                   of how to specify other sets of classes. 
                   TYPE: string
                   DEFAULT: 'digits'
        - num_images: Number of images over which to test.
                              TYPE: int
                              DEFAULT: If num_images is not specified, the test will be over all the test images of
                                       of the EMNIST dataset for the chosen split and classes.
        - num_examples: Number of "examples" images from each class used to classify test images by comparison. These
                        images are taken from the EMNIST training data.
    """
    
    # -------------------------- SETTING UP --------------------------
    ### DEFINITIONS
    # Folder
    required_files = ['network_state_dict.pth', 'network_info.pkl']
    if not folder_tool.check_folder(folder, required_files):
        raise OSError("Given folder does not contain a network state dict to test.")
    
    # Device
    if 'device' in kwargs.keys():
        device = kwargs['device']
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    
    # Dataloader. batch_size=None by default, so we load one element at a time.
    split          = kwargs['emnist_split'] if ('emnist_split' in kwargs) else 'byclass'
    classes_string = kwargs['classes']      if ('classes'      in kwargs) else 'digits'
    num_images     = kwargs['num_images']   if ('num_images'   in kwargs) else -1
    loader = dataloaders.EmnistSinglesLoader(data_path = data_path,
                                             split = split,
                                             classes_string = classes_string,
                                             num_images = num_images,
                                             inds_save_folder = folder,
                                             device = device) 
    num_images = loader.num_images
    
    # Classes dictionary
    class_dict = {}
    class_labels = loader.class_labels
    classes = loader.classes
    num_classes = len(class_labels)
    for i in range(len(class_labels)):
        class_dict[str(i)] = (class_labels[i], loader.classes[i])
    
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
    softmax = torch.nn.Softmax(dim=1)

    
    ### GET EXAMPLE FEATURES
    # Examples images from taken from either from the training set or from the memory.pkl file. Classification of test
    # data is performed by evaluating the similarity of individual test images with example images from each class.
    use_mem = ('use_memory' in kwargs) and kwargs['use_memory']
    if use_mem:
        # Load memory, check if all required classes are present
        mem_file = os.path.join(folder, 'memory.pkl')
        assert os.path.isfile(mem_file), f"'memory.pkl' missing from {folder}"
        with open(mem_file, 'rb') as f:
            ex_feats_dict_ = pickle.load(f) # Dictionary of examples of features corresponding to each class
        er_message  = "At least one required class is missing from registered memory.\n"
        er_message += "Either set use_memory=False, or register all classes specified by classes_string before runnning classification_test."
        assert all(class_ in ex_feats_dict_.keys() for class_ in classes), er_message
        # Get examples of features for required classes and feature blocks.
        blocks = [str(b) for b in blocks]
        ex_feats_dict = {}
        for class_ in classes:
            class_features = {}
            for block in blocks:
                class_features[block] = ex_feats_dict_[class_][block].to(device)
            ex_feats_dict[class_] = class_features
        # Find minimum number of registered elements for all classes and all feature blocks. Check it is > 0.
        num_examples = min(min(feat.shape[0] for feat in class_feat_dict.values()) for class_feat_dict in ex_feats_dict_.values())
        assert num_examples > 0, "At least one class has no registered memory for at least one feature block."  
    else: 
        # Find images from each class in data and map them to features.
        train_ds = dataloaders.emnist(data_path = data_path, split = split, train = True)
        num_examples = kwargs['num_examples'] if ('num_examples' in kwargs) else 10
        examples = emnist_tools.examples_assembly(ds = train_ds, 
                                                  class_labels = class_labels, 
                                                  num_examples = num_examples, 
                                                  shuffled = True,
                                                  device = device)
        ex_feats_dict = {} # Examples of features corresponding to each class
        for i in range(num_classes):
            ex_feats_dict[classes[i]] = net.feature_network(examples[i])

    
    ### LOGGING
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
    
    message  = f"CLASSIFICATION TEST on the following classes:\n{loader.classes}\n"
    message += f"Will test over a total of {num_images} images.\n"
    message += f"Number of example images per class used for classifying test data: {num_examples} (taken from training data).\n"
    message += f"EMNIST split = {split}\n"
    message +=  "\n...\n"
    logger.log(2, message)
    


    # -------------------------- CLASSIFYING --------------------------
    total    = [0]*num_classes  # Records number of images seen from each class
    ave_rank = [0]*num_classes  # Records average rank of correct class in predictions
    pred_count = torch.zeros((num_classes, num_classes))  # Entry [i,j] will record the pnumber of images from the i^th
                                                          # class that are classified to the j^th class by the network.
    ave_scores = torch.zeros((num_classes, num_classes))  # Entry [i,j] will record average prediction score of the j^th
                                                          # class when the input is from the i^th class. The ideal scenario
                                                          # is when this is an identity matrix.
    time_start = time.perf_counter()
    with torch.no_grad():
        for im, label in tqdm(loader):
            ### CLASSIFY
            # Get image, label and map image to features
            im = im.to(device)
            im_label_ind = class_labels.index(label)
            im_stack = torch.stack([im]*num_examples) # Duplicate to fit the number of images stored in memory for each class.
            im_feat_stack = net.feature_network(im_stack)
            # Compute score for each class and class prediction
            scores = []
            for i in range(num_classes):
                scores.append(torch.mean(softmax(net.evaluator(im_feat_stack, ex_feats_dict[classes[i]]))[:,1].cpu().data))
            scores = torch.stack(scores)

            ### KEEP TRACk OF RESULTS
            total[im_label_ind] += 1
            ave_scores[im_label_ind] += scores  # Record full scores
            pred = int(torch.argmax(scores)) # Prediction
            pred_count[im_label_ind, pred] += 1
            correct_class_rank = torch.where(scores >= scores[im_label_ind])[0].shape[0] # Rank of correct class within predictions
            ave_rank[im_label_ind] += correct_class_rank 
   
    correct = []
    pred_props = torch.zeros((num_classes, num_classes))
    freq_guesses = []
    for i in range(num_classes):
        correct.append(int(pred_count[i,i]))
        ave_scores[i] /= total[i]
        ave_rank[i] /= total[i]
        pred_props[i] = pred_count[i]/total[i]
        freq_guesses.append(classes[int(pred_count[i].argmax())])
    ave_rank = [round(float(ave_rank[i]), 1) for i in range(num_classes)]
    correct_ovrl = int(sum(correct)) # Overall correct guesses
    correct_ovrl_pct = round(float(correct_ovrl/num_images * 100), 2)
    correct_pcts = [round(float(correct[i]/total[i]*100), 2) for i in range(num_classes)] # Correct guesses per class 
        
    classification_time = time.perf_counter() - time_start
       

    # -------------------------- EPILOGUE --------------------------
    samples_per_sec = num_images/classification_time
    message  = f"Testing done after {int(classification_time)} seconds\n({samples_per_sec:.3f} samples per second.)\n\n"
    message += "RESULTS TABLE\n"
    message += "(Average rank: Average rank of correct class in network's classifications, for given class.)\n"
    message += "(Most frequent: Most frequent class guessed by network for images of given class.)\n"
    
    
    # Correct guesses per class 
    correct_pcts = [round(float(correct[i]/total[i]*100), 2) for i in range(num_classes)]
    
    # Make table 1
    row_names = [[''],
                 ['Total guesses'],
                 ['Correct guesses'],
                 ['Correct %'],
                 ['Average rank\n'],
                 ['Most frequent']]
    results_full = [['Overall'] + classes,
                    [num_images] + total,
                    [correct_ovrl] + correct,
                    [correct_ovrl_pct] + correct_pcts,
                    ['-'] + ave_rank,
                    ['-'] + freq_guesses]
    
    num_rows = (num_classes + 1) // 15 + 1 # Split table to avoid making it too wide.
    for i in range(num_rows):
        low_ind, up_ind = i*15, (i+1)*15
        results = [r[low_ind:up_ind] for r in results_full]
        table = []
        for j in range(6):
            table.append(row_names[j] + results[j])
        message += tabulate(table, headers="firstrow", tablefmt="grid")
        message += '\n\n'
    
    # Make table 2
    message += "FULL CLASSIFICATION RESULTS. In the following table, the entry in the i^th row and j^th column\n"
    message += "represents the proportion of images from the j^th class that were classified in the i^th class by\n"
    message += "the network. (Ideally, this table looks like an identity matrix.)\n"
    pred_props = pred_props.numpy().swapaxes(0,1)
    for i in range(num_rows):
        low_ind, up_ind = i*15, min((i+1)*15, num_classes)
        message += tabulate(pred_props[:,low_ind:up_ind], classes[low_ind:up_ind], showindex=classes, tablefmt="grid", floatfmt=".3f")
        message += '\n\n'
    
    # Make space for next time, log message.
    message +=  "\n\n\n\n----------------------\n\n\n"
    logger.log(2, message)
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser('...')
    # General
    parser.add_argument('--folder',       type=str,  required=True, help="Folder containing the SymbolNet network to test.")
    parser.add_argument('--num_images',   type=int,  default=-1,    help="Number of samples over which to test. Each sample is one pair of images.")
    parser.add_argument('--num_examples', type=int,  default=10,    help="If not using memory, number of examples from each class used to classify new images.")
    parser.add_argument('--use_memory',   action='store_true',      help="Load feature examples from memory file")
    # Data
    parser.add_argument('--data_path',    type=str, required=True,      help="Where the data is")
    parser.add_argument('--emnist_split', type=str, default='balanced', help="Which split of the EMNIST dataset to use")
    parser.add_argument('--classes',      type=str, default='digits',   help="String specifying which ")
    # Network
    parser.add_argument('--architecture', default='SymbolNet',   choices=['SymbolNet'])
    parser.add_argument('--compare_mode', default='subtract', choices=['subtract', 'concatenate'])
    parser.add_argument('--blocks',       type=str, default='1,2,3', help='Which feature network block outputs to use in evaluation.')
    
    args = parser.parse_args()
    classification_test(**args.__dict__)
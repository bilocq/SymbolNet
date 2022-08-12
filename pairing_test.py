"""
Testing a SymbolNet on whether it correctly classifies pairs of images as being from the same class
or from different classes.
"""

import os
import time
import argparse
import logging
import pickle
from tqdm import tqdm
import torch

from symbolnet_utils import dataloaders, networks, folder_tool



def pairing_test(folder, data_path, num_samples, **kwargs):
    """
    Loads the state dictionary of a network from symbolnet_utils.networks and test its ability to recognize whether 
    both images in input pairs are from the same class or from different classes. In this form, this function is
    specifically suited to the EMNIST dataset, as it relies on the EmnistPairsLoader from symbolnet_utils.dataloaders.
    
    Note that the classes used for testing do not need to be the same as the classes that were used for training 
    beforehand!
    
    REQUIRED PARAMETERS
        - folder: A folder containing the state dictionary to test, under folder/network_state_dict.pth
        - data_path: Path where the EMNIST dataset is located. The data_path path must lead to a folder containing
                     a folder called EMNIST which itself contains the dataset. If the dataset is not there, then
                     it gets downloaded automatically. 
        - num_samples: Number of samples over which to test. Each sample is one pair of images.
    
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
        - batch_size: Batch size used for testing.
                      TYPE: int
                      DEFAULT: 64
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
        
    
    # Dataloader
    split          = kwargs['emnist_split']   if ('emnist_split'   in kwargs) else 'byclass'
    classes_string = kwargs['classes']        if ('classes'        in kwargs) else 'digits'
    batch_size     = kwargs['batch_size']     if ('batch_size'     in kwargs) else 64
    num_samples = num_samples - (num_samples % batch_size) # Make num_samples a multiple of batch_size
    loader = dataloaders.EmnistPairsLoader(data_path = data_path,
                                           split = split,
                                           classes_string = classes_string,
                                           batch_size = batch_size,
                                           inds_save_folder = folder,
                                           device = device,
                                           train=False)
    
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
    
    message  = f"PAIR RECOGNITION TEST on the following classes:\n{loader.classes}\n"
    message += f"Will test over a total of {num_samples} samples.\n"
    message += f"EMNIST split = {split}\n"
    message +=  "\n...\n"
    logger.log(2, message)
    
    
    # -------------------------- TESTING --------------------------
    num_batches = num_samples//batch_size
    
    total0 = 0 # Counter for the number of samples labelled 0 (i.e. both images from same class)
    correct0 = 0 # Counter for the number of samples labelled 0 and correctly classified by net
    margin0 = 0 # Will be incremented by logits margin in favor of logit 0 on samples labelled 0
    total1 = 0 # Same thing for samples labelled 1 (i.e. images from different classes) 
    correct1 = 0
    margin1 = 0

    time_start = time.perf_counter()
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            b0, b1, t = loader()
            out = softmax(net(b0, b1))
            
            # Batch classification statistic
            inds0 = torch.nonzero(t==0).view(-1) # Indices of samples labelled 0 (i.e. both images from same class) within batch
            total0 += inds0.shape[0] # Total number of samples labelled 0 in batch
            correct0 += torch.nonzero(out[inds0,0] > out[inds0,1]).shape[0] # Number of those samples that are correctly classified
            margin0 += torch.sum(out[inds0,0] - out[inds0,1]) # Average correct classification logits margin
            
            inds1 = torch.nonzero(t==1).view(-1) # Same thing for samples labelled 1 (i.e. images from different classes)
            total1 += inds1.shape[0]
            correct1 += torch.nonzero(out[inds1,1] > out[inds1,0]).shape[0] 
            margin1 += torch.sum(out[inds1,1] - out[inds1,0]) 
    
    testing_time = time.perf_counter() - time_start
    
    # -------------------------- EPILOGUE --------------------------
    samples_per_sec = num_samples/testing_time
    
    correct_pct0 = correct0/total0 * 100
    ave_margin0 = margin0/total0
    correct_pct1 = correct1/total1 * 100
    ave_margin1 = margin1/total1
    correct_tot = correct0 + correct1
    correct_pct_tot = (correct_tot)/num_samples * 100
    ave_margin_tot = (margin0 + margin1)/num_samples
    
    message  = f"Testing done after {int(testing_time)} seconds\n({samples_per_sec:.3f} samples per second.)\n\n"
    message +=  "RESULTS\n"
    message += f"{correct_pct_tot:.2f}% samples correctly classified overall ({correct_tot} correct out of {num_samples})\n"
    message += f"{correct_pct0:.2f}% of samples with images from same class correctly classified ({correct0} correct out {total0})\n"
    message += f"{correct_pct1:.2f}% of samples with images from different classes correctly classified ({correct1} correct out {total1})\n"
    message += f"Total average margin: {ave_margin_tot:.2f}\n"
    message += f"Average margin for samples with images from different classes: {ave_margin0:.2f}\n"
    message += f"Average margin for samples with images from the same class: {ave_margin1:.2f}."
    message +=  "\n\n\n\n----------------------\n\n\n"
    logger.log(2, message)
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('...')
    # General
    parser.add_argument('--folder',       type=str,  required=True, help="Folder containing the SymbolNet network to test.")
    parser.add_argument('--num_samples',  type=int,  required=True, help="Number of samples over which to test. Each sample is one pair of images.")
    # Data
    parser.add_argument('--data_path',    type=str, required=True,     help="Where the data is")
    parser.add_argument('--emnist_split', type=str, default='balanced', help="Which split of the EMNIST dataset to use")
    parser.add_argument('--classes',      type=str, default='digits',  help="String specifying which ")
    # Network
    parser.add_argument('--architecture', default='SymbolNet',   choices=['SymbolNet'])
    parser.add_argument('--compare_mode', default='subtract', choices=['subtract', 'concatenate'])
    parser.add_argument('--blocks',       type=str, default='1,2,3', help='Which feature network block outputs to use in evaluation.')
    # Testing
    parser.add_argument('--batch_size',   type=int,  default=100)
    
    args = parser.parse_args()
    pairing_test(**args.__dict__)
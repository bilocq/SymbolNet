"""
Main training function for SymbolNet
"""

import os
import time
import argparse
import logging
import pickle
from tqdm import tqdm
import torch
from torch import nn

from symbolnet_utils import dataloaders, networks, training_stats, folder_tool, emnist_tools



def train(folder, data_path, num_samples, **kwargs):
    """
    Trains a network from symbolnet_utils.networks to recognize whether both of its image inputs are from the same class
    or from different classes. In this form, this function is specifically suited to the EMNIST dataset, as it relies
    on the EmnistPairsLoader from symbolnet_utils.dataloaders.
    
    REQUIRED PARAMETERS
        - folder: A folder where results and state dictionaries are saved at the end and potentially loaded at the 
                  start. If the folder already exists and contains the required files, training resumes from where
                  it was saved; otherwise it starts from scratch. See symbolnet_utils.folder_tool for specific behavior.
        - data_path: Path where the EMNIST dataset is located. The data_root path must lead to a folder containing
                     a folder called EMNIST which itself contains the dataset. If the dataset is not there, then
                     it gets downloaded automatically.   
        - num_samples: Number of samples over which to train. Each sample is one pair of images.
    
    OPTIONAL PARAMETERS
        - device: Can be either torch.device('cpu') or torch.device('cuda').
                  TYPE: torch.device  
                  DEFAULT: torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        - emnist_split: Which split of the EMNIST dataset to use. Can be either 'byclass', 'bymerge', 'balanced'  
                        or 'digits' (the 'letters' split is not supported).
                        TYPE: string
                        DEFAULT: 'byclass'
        - classes: A string specifying which classes to use for training. Basic options are 'digits', 'uppercases'
                   and 'lowercases'. See symbolnet_utils.emnist_tools.EmnistClassLabelsSetup for a detailed explanation
                   of how to specify other sets of classes.
                   TYPE: string
                   DEFAULT: 'digits'
        - batch_size: Batch size used for training.
                      TYPE: int
                      DEFAULT: 64
        - lr: Learning rate
              TYPE: float
              DEFAULT: 0.001
        - beta_1: First beta parameter for the Adam optimizer. 
                  TYPE: float 
                  DEFAULT: 0.9
        - beta_2: Second beta parameter for the Adam optimizer.
                  TYPE: float
                  DEFAULT: 0.999
        - weight_decay: Weight decay parameter for the optimizer. 
                        TYPE: float
                        DEFAULT: 0.0001.
        - verbose_training: If true, statistics of classification over latest training batch get printed to
                            console at certain points during training (up to twenty times).
                            TYPE: bool
                            DEFAULT: False
    """
    
    # -------------------------- SETTING UP --------------------------
    ### FOLDER
    required_files = ['network_state_dict.pth', 'network_info.pkl', 
                      'optimizer_state_dict.pth', 'log.txt', 'training_stats.pkl']
    resume = folder_tool.check_folder(folder, required_files)
    
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
    
    
    ### DEFINITIONS
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
                                           device = device)
    
    # Network information file
    if not resume: # Get network settings from kwargs or default, and save to file
        arch = kwargs['architecture'] if ('architecture' in kwargs) else 'SymbolNet'
        comp_mode = kwargs['compare_mode'] if ('compare_mode' in kwargs) else 'subtract'
        blocks = kwargs['blocks'] if ('blocks' in kwargs) else '1,2,3'
        
        start_sample = 0
        last_sample  = num_samples
        net_dict = {'architecture'         : arch, 
                    'compare_mode'         : comp_mode, 
                    'blocks'               : blocks, 
                    'num_training_samples' : num_samples,
                    'training_classes'     : emnist_tools.sort_classes(loader.classes)}
        with open(os.path.join(folder, 'network_info.pkl'), 'wb') as net_dict_file:
            pickle.dump(net_dict, net_dict_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    else: # Load network settings from file
        with open(os.path.join(folder, 'network_info.pkl'), 'rb') as net_dict_file:
            net_dict = pickle.load(net_dict_file)
        arch      = net_dict['architecture']
        comp_mode = net_dict['compare_mode']
        blocks    = net_dict['blocks']
        
        # Issue warning if there is a discrepancy with command line settings
        for setting in ['architecture', 'compare_mode', 'blocks']:
            if (setting in kwargs) and kwargs[setting] != net_dict[setting]:
                message  = f"WARNING: The network setting '{setting}' given as argument is different"
                message +=  " from the what was saved in network_info.pkl. We default to the"
                message +=  " saved value so that the network state dictionary can be reloaded."
                logger.log(1, message)
        
        # Update network training info
        start_sample = net_dict['num_training_samples']
        last_sample  = start_sample + num_samples
        net_dict['num_training_samples'] = last_sample
        all_training_classes = emnist_tools.sort_classes(net_dict['training_classes'] + loader.classes)
        net_dict['training_classes'] = all_training_classes
        with open(os.path.join(folder, 'network_info.pkl'), 'wb') as net_dict_file:
            pickle.dump(net_dict, net_dict_file, protocol=pickle.HIGHEST_PROTOCOL)
     
    # Network
    arch   = getattr(networks, arch)
    blocks = [int(b) for b in blocks.split(',')]
    net = arch(blocks=blocks, compare_mode=comp_mode).to(device) 
    
    # Loss function, optimizer
    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1) # Used to record training statistics
    lr           = kwargs['lr']           if ('lr'           in kwargs) else 0.001
    beta_1       = kwargs['beta_1']       if ('beta_1'       in kwargs) else 0.9
    beta_2       = kwargs['beta_2']       if ('beta_2'       in kwargs) else 0.999
    weight_decay = kwargs['weight_decay'] if ('weight_decay' in kwargs) else 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr , betas=(beta_1, beta_2), weight_decay=weight_decay)

    # A few more things to do if resuming
    if resume:
        net.load_state_dict(torch.load(os.path.join(folder, 'network_state_dict.pth'), map_location=device))
        optimizer.load_state_dict(torch.load(os.path.join(folder, 'optimizer_state_dict.pth')))
        training_stats.load(folder)
        training_stats.offset(start_sample)
        
    
    ### INITIAL MESSAGE
    if not resume: # Start from scratch
        message  = f"TRAINING FROM SCRATCH on the following classes:\n{loader.classes}\n"
        message += f"EMNIST split = {split}\n"
        message += f"Will train up to {num_samples} samples\n"
        
    else: # Resume from previous training
        message  =  "TRAINING - RESUMING FROM PRIOR TRAINING (Reloading network and optimizer state dictionary.)\n"
        message += f"Now training on the following classes:\n{loader.classes}\n"
        message += f"Will train up to {last_sample} samples.\n"
        message += f"EMNIST split = {split}\n"

    message += f"Batch size: {batch_size}\n"
    message += f"lr: {lr}   betas: {(beta_1, beta_2)}   weight_decay: {weight_decay}\n" 
    message += f"Device: {device}\n"
    message +=  "\n...\n"
    logger.log(2, message)


    # -------------------------- TRAINING --------------------------
    num_batches = num_samples//batch_size
    verbose_training = ('verbose_training' in kwargs) and kwargs['verbose_training'] and (num_batches >= 20)
    samples_done = 0
    time_start = time.perf_counter()
    for i in tqdm(range(num_batches)):
        # Training step
        optimizer.zero_grad()
        b0, b1, t = loader()
        out = net(b0, b1)
        loss = loss_fn(out, t)
        loss.backward()
        optimizer.step()
        samples_done += out.shape[0]
        
        # Batch classification statistics
        out = softmax(out) # Makes margins more meaningful, doesn't affect the rest.
        
        inds0 = torch.nonzero(t==0).view(-1) # Indices of samples labelled 0 (i.e. both images from same class)
        correct0 = torch.nonzero(out[inds0,0] > out[inds0,1]).shape[0] # Number of those samples that are correctly classified
        prop0 = correct0/inds0.shape[0] # Proportion correctly classified
        margin0 = torch.mean(out[inds0,0] - out[inds0,1]) # Average correct classification logits margin
        
        inds1 = torch.nonzero(t==1).view(-1) # Same thing for samples labelled 1
        correct1 = torch.nonzero(out[inds1,1] > out[inds1,0]).shape[0] 
        prop1 = correct1/inds1.shape[0] 
        margin1 = torch.mean(out[inds1,1] - out[inds1,0]) 
        prop_ovrl = (correct0 + correct1)/out.shape[0] # Overall proportion of correctly classified samples over batch
        
        training_stats.tick(out.shape[0])
        training_stats.plot("Correct guess same class proportion", prop0)
        training_stats.plot("Correct guess diff classes proportion", prop1)
        training_stats.plot("Correct guess overall proportion", prop_ovrl)
        training_stats.plot("Classification margin same class", margin0.cpu().data.numpy())
        training_stats.plot("Classification margin diff classes", margin1.cpu().data.numpy())
        
        if verbose_training and (i % (1+ num_batches//20) == 0):
            message  = f"After {samples_done} samples, classification results on most recent batch:\n"
            message += f"Correct guess same class proportion: {prop0:.4f}\n"
            message += f"Correct guess diff classes proportion: {prop1:.4f}\n"
            message += f"Correct guess overall proportion: {prop_ovrl:.4f}\n"
            message += f"Classification margin same class: {margin0:.4f}\n"
            message += f"Classification margin diff classes: {margin1:.4f}\n"
            logger.log(1, message)
    
    training_time = time.perf_counter() - time_start
            
    
    # -------------------------- EPILOGUE --------------------------
    samples_per_sec = num_samples/training_time
    training_stats.flush(folder)
    
    torch.save(net.state_dict(), os.path.join(folder, 'network_state_dict.pth'))
    torch.save(optimizer.state_dict(), os.path.join(folder, 'optimizer_state_dict.pth'))
    
    message  = f"DONE AFTER {int(training_time)} seconds\n({samples_per_sec:.3f} samples per second)"
    message +=  "\n\n\n\n----------------------\n\n\n"
    logger.log(2, message)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('...')
    # General
    parser.add_argument('--folder',       type=str,  required=True, help="Folder to create or from which to resume")
    parser.add_argument('--num_samples',  type=int,  required=True, help="Number of samples over which to train. Each sample is one pair of images.")
    # Data
    parser.add_argument('--data_path',    type=str, required=True,     help="Where the data is.")
    parser.add_argument('--classes',      type=str, default='digits',  help="String specifying which classes to use.")
    parser.add_argument('--emnist_split', type=str, default='byclass', help="Which split of the EMNIST dataset to use.")
    # Network
    parser.add_argument('--architecture', default='SymbolNet',   choices=['SymbolNet'])
    parser.add_argument('--compare_mode', default='subtract', choices=['subtract', 'concatenate'])
    parser.add_argument('--blocks',       type=str, default='1,2,3', help='Which feature network block outputs to use in evaluation.')
    # Training
    parser.add_argument('--batch_size',   type=int,   default=100)
    parser.add_argument('--lr',           type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--verbose_training', action='store_true', help="Print classification statistics of some training batches?")
    
    args = parser.parse_args()
    train(**args.__dict__)
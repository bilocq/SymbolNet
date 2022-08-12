"""
SymbolNet neural network architecture.
"""

import numpy as np
import torch
import torch.nn as nn




class feature_network(nn.Module):
    """
    The feauture network maps images to corresponding sets of deep features. It is composed of either 1, 2 or 3
    convolutional 'blocks'. The output is a dictionary of features with keys corresponding to the block indices 
    specified by the 'out_blocks' setting (the keys are in string format). The blocks are in consecutive order
    and may be part of the feature network even without being specified in 'out_blocks', specifically if there's
    a subsequent block specified in 'out_blocks'.
    
    EXAMPLES:
        - out_blocks==[1]: The feature network is composed of one block only, i.e. block1, and the output
          features dictionaries have only one key, i.e. '1'.
        - out_blocks==[2]: The feature network is composed of two blocks, i.e. block1 followed by block2, and
          the output features dictionaries have only one key, i.e. '2'.
        - out_blocks==[1,2,3]: The feature network is composed of all three blocks, i.e. block1 followed by
          block2 followed by block3, and the output features dictionaries have the three keys '1', '2', '3'.
    """
    
    def __init__(self, in_channels=1, out_blocks=[1,2,3]):
        super().__init__()
        
        if isinstance(out_blocks, int):
            out_blocks = [out_blocks]
        
        max_block = max(out_blocks)
        block1 = nn.Sequential(nn.Conv2d(in_channels, 20, kernel_size=5),
                               nn.ReLU(),
                               nn.Conv2d(20, 40, kernel_size=5),
                               nn.ReLU(),
                               nn.Conv2d(40, 60, kernel_size=3),
                               nn.ReLU())
        block_list = [['1', block1]]
        if max_block > 1:
            block2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   nn.Conv2d(60, 80, kernel_size=3),
                                   nn.ReLU(),
                                   nn.Conv2d(80, 100, kernel_size=3),
                                   nn.ReLU())
            block_list.append(['2', block2])
        if max_block > 2:
            block3 = nn.Sequential(nn.Conv2d(100, 150, kernel_size=3),
                                   nn.ReLU(),
                                   nn.Conv2d(150, 200, kernel_size=3),
                                   nn.ReLU())
            block_list.append(['3', block3])
            
        self.blocks = nn.ModuleDict(block_list)
        self.out_blocks = out_blocks
        
    def forward(self, x):
        features = {}
        for i in range(1,4):
            x = self.blocks[str(i)](x)
            if i in self.out_blocks:
                features[str(i)] = x
            if i == max(self.out_blocks):
                break
        return features



class evaluator(nn.Module):
    """
    The evaluator maps pairs of sets of features - each set of features being obtained by mapping an image through 
    the feature network, to outputs containing 2 unnormalized logits. For a given pair of sets of features, the
    first logit being larger indicates the evaluator thinks the two images are from different classes, while the
    second logit being larger indicates the evaluator thinks the images are from the same class.
    
    The be matched with a feature network, the 'in_blocks' setting must be the same are that feature network's
    'out_blocks' setting.
    
    Each block of feature pairs of the input is first mapped separately through a dedicated fully connected layer.
    The outputs of these fully connected layers are then concatenated and mapped through the final layers together.
    
    The 'compare_mode' argument determines how the the pairs of features are combined in each input block. The two
    options are: 'subtract' (the difference of the pair of feature tensors is the input of the block's dedicated 
    fully connected layer) and 'concatenate' (the concatenation of the pair of feature tensors is the input of the
    block's fully connected layer).
    """
    
    def __init__(self, im_size=(28,28), in_blocks=[1,2,3], compare_mode='subtract'):
        super().__init__()
        
        # Sanity checks
        assert all(s >= 28 for s in im_size), "Requested input size too small. Minimum size is 22 in each dimension."
        if isinstance(in_blocks, int):
            in_blocks = (in_blocks)
        assert all((0<b<4 and isinstance(b,int)) for b in in_blocks), "Requested invalid blocks. Valid block labels are 1, 2, 3 and ints."
        assert compare_mode in ['subtract', 'concatenate'], f"Invalid compare_mode {compare_mode}"
        
        self.in_blocks = in_blocks
        self.compare_mode = compare_mode
        
        # Calculate width and height of outputs of each block
        size1 = [s - 10 for s in im_size]
        size2 = [int(s/2 - 4) for s in size1]
        size3 = [s-4 for s in size2]
        
        # Calculate total size of outputs of each block
        size1 =  60 * np.prod(size1)
        size2 = 100 * np.prod(size2)
        size3 = 200 * np.prod(size3)
        sizes = {'1' : size1, '2' : size2, '3' : size3}
        
        # First layer is separate for each block
        block_layers = []
        in_multiplier = 1 if compare_mode == 'subtract' else 2
        for i in in_blocks:
            block = nn.Sequential(nn.Linear(sizes[str(i)]*in_multiplier, 50), nn.ReLU())
            block_layers += [str(i), block],
        self.block_layers = nn.ModuleDict(block_layers)
        
        # Last layers
        self.final_layers = nn.Sequential(nn.Linear(50*len(in_blocks), 50),
                                          nn.ReLU(),
                                          nn.Linear(50, 2))
        
    def forward(self, features1, features2):
        deep_features = []
        for i in features1.keys():
            if self.compare_mode == 'subtract':
                features = (features1[i] - features2[i]).view(features1[i].shape[0], -1)
                deep_features.append(self.block_layers[i](features))
            elif self.compare_mode == 'concatenate':
                f1 = features1[i].view(features1[i].shape[0], -1)
                f2 = features2[i].view(features1[i].shape[0], -1)
                features = torch.cat((f1, f2), dim=1)
                deep_features.append(self.block_layers[i](features))
            
        deep_features = torch.cat(deep_features, dim=1)
        logits = self.final_layers(deep_features)
        
        return logits
        
                

class SymbolNet(nn.Module):
    """
    The SymbolNet architecture combines a feature network and an evaluator to map a pair of input images to a pair of 
    unnormalized logits indicating whether the network thinks the images are from the same class or from different
    classes.
    
    PARAMETERS:
        - in_channels: Number of color channels in the input images. Default is 1 (i.e. grayscale) since EMNIST
                       images  are grayscale.
                       TYPE: int
                       DEFAULT: 1
        - im_size: Pixel size of the input images. Default is (28,28) because that's the size of EMNIST images
                   TYPE: tuple of 2 ints
                   DEFAULT: (28,28)
        - blocks: Which feature blocks to use as outputs of the feature networks and inputs of the evaluator. See
                  explanations above under 'feature_network' and 'evaluator'.
                  TYPE: A list containing some of or all of the integers 1, 2 and 3.
                  DEFAULT: [1,2,3]
    """
    
    def __init__(self, in_channels=1, im_size=(28,28), blocks=[1,2,3], compare_mode='subtract'):
        super().__init__()
        self.feature_network = feature_network(in_channels=in_channels, out_blocks=blocks)
        self.evaluator = evaluator(im_size=im_size, in_blocks=blocks, compare_mode=compare_mode)
        # For easy access later:
        self.architecture = 'SymbolNet'
        self.compare_mode = compare_mode
        self.blocks = ','.join([str(b) for b in blocks]) # Gives '1,2,3' if blocks==[1,2,3]

    def forward(self, x1, x2):
        features1 = self.feature_network(x1)
        features2 = self.feature_network(x2)
        return self.evaluator(features1, features2)
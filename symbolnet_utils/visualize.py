"""
Tools to visualize the inputs and outputs of SymbolNet
"""

from PIL import ImageDraw
import torch
from torchvision import transforms, utils

from symbolnet_utils import dataloaders, networks
from emnist_tools import example_assembly



def visualize_pairs(network, batch=None, data_path=None, **kwargs):
    """
    Creates an image which shows pairs of images and corresponding similarity values. The similarity value
    is a number between 0 and 1, with anything below 0.5 indicating the network thinks the images are from 
    different classes and anything above 0.5 indicating the the network thinks the images are from the same
    class. The similarity values appear with either a green or red background depending on whether the resulting
    classification is correct.
    
    REQUIRED PARAMETER:
        - network: An instance of the networks.SymbolNet class.
    
    OPTIONAL PARAMETERS:
        - device: Can be either torch.device('cpu') or torch.device('cuda').
                  TYPE: torch.device  
                  DEFAULT: torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        - batch: (torch.tensor or None) A batch from an instance of dataloaders.EmnistPairsLoader
                 DEFAULT: None
        - data_path: (string or None) Path to EMNIST dataset. If dataset isn't already there, it gets downloaded.
                     DEFAULT: None
        - num_samples: (int) The number of pair of images to include in the output. 
                       DEFAULT: 10
        - emnist_split: (string) Which split of the EMNIST dataset to use.
                        DEFAULT: 'balanced'
        - classes_string: (string) A string specifying the classes to be used. See emnist_tools.EmnistClassLabelsSetup
                          for a detailed explanationof how to specify sets of classes.
                          DEFAULT: 'digits'
        
    NOTE:
        Either the 'batch' or the 'data_path' argument must be specified. If 'batch' is specified, then all other 
        optional parameters are ignored.
    """
    # Device
    if 'device' in kwargs.keys():
        device = kwargs['device']
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Network
    if type(network) == str: # Load network state dictionary from path
        net = networks.SymbolNet()
        net.load_state_dict(torch.load(network, map_location=device))
    else: # Else we assume network is already loaded with correct state dictionary
        net = network
    softmax = torch.nn.Softmax(dim=1)
    
    # Transforms for creating image
    to_im = transforms.ToPILImage()
    to_tens = transforms.ToTensor()
    
    # Data
    if batch is not None:
        b0, b1, t = batch
    elif data_path is not None:
        num_samples    = kwargs['num_samples']    if ('num_samples'    in kwargs) else 10
        split          = kwargs['emnist_split']   if ('emnist_split'   in kwargs) else 'balanced'
        classes_string = kwargs['classes_string'] if ('classes_string' in kwargs) else 'digits' 
        loader = dataloaders.EmnistPairsLoader(data_path, split, classes_string, num_samples, device, train=False)
        b0, b1, t = loader()
    else:
        raise ValueError("Either the 'batch' or the 'data_root' argument must be specified.")
    
    # Similarity predictions and corresponding boxes in image
    num_samples = b0.shape[0]
    out = softmax(network(b0, b1))
    simboxes = []
    for i in range(num_samples):
        if torch.argmax(out[i]) == t[i]: # Correctly classified
            color = torch.tensor([0,1,0]) # green
        else:                            # Wrongly classified
            color = torch.tensor([1,0,0]) # red
        # Make uniform color box
        simbox = color[:,None,None].expand(-1, b0.shape[2], b0.shape[3]).float()
        simbox = to_im(simbox)
        # Write similarity value in box
        simtext = f'{out[i,1]:.2f}'
        simbox_edit = ImageDraw.Draw(simbox)
        simbox_edit.text((3,9), simtext, (0,0,0))
        # Turn box back into tensor so we can easily glue things together conveniently
        simbox = to_tens(simbox)
        simboxes.append(simbox)
    simboxes = torch.stack(simboxes, dim=0) # num_samples x 3 x b0.shape[2] x b0.shape[3]
    
    # Create image
    b0 = b0.expand(-1,3,-1,-1)
    b1 = b1.expand(-1,3,-1,-1)
    buffer0 = torch.ones((num_samples, b0.shape[1], b0.shape[2], 2))*0.5 # Grey vertical stripes
    pair_batch = torch.cat((b0, buffer0, b1, buffer0, simboxes), dim=3)
    buffer1 = torch.ones((num_samples, pair_batch.shape[1], 1, pair_batch.shape[3])) # White horizontal lines 
    pair_batch = torch.cat((pair_batch, buffer1), dim=2)
    image_tensor = utils.make_grid(pair_batch, nrow = 1, padding = 0, pad_value = 0.9).cpu()
    image_tensor = image_tensor[:,0:-1,:] # Remove last white line
    image = to_im(image_tensor)
    
    # Save
    if 'save_path' in kwargs:
        image.save(kwargs['save_path'])
    
    # Show image
    if ('show_image' in kwargs) and bool(kwargs['show_image']):
        image.show()
    
    return image, image_tensor
 
    
    
def visualize_classification(network, image, data_path, **kwargs):
    """
    Creates an image by glueing together several images from 'visualize_pairs', but where the first image is
    always the same and it gets compared to images from different classes. This is reminiscent of what happens
    when classifying the first image with SymbolNet. Each column in the resulting image corresponds to comparing 
    the first image to samples from a given class.
    
    REQUIRED PARAMETERS:
        - network: An instance of the networks.SymbolNet class.
        - image: (torch.tensor) an (unbatched) image to be compared to samples from different classes.
        - data_path: (string or None) Path to EMNIST dataset. If dataset isn't already there, it gets downloaded.
                     DEFAULT: None
    
    OPTIONAL PARAMETERS:
        - device: Can be either torch.device('cpu') or torch.device('cuda').
                  TYPE: torch.device  
                  DEFAULT: torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        - num_examples: (int) Number of examples from each class to compare to 'image'
        - emnist_split: (string) Which split of the EMNIST dataset to use.
                        DEFAULT: 'balanced'
        - classes_string: (string) A string specifying to which classes 'image' should be compared. See 
                          emnist_tools.EmnistClassLabelsSetup for a detailed explanationof how to specify 
                          sets of classes.
                          DEFAULT: 'digits'
        - label: Class label of the 'image' input. If this is not specified, the class of the image is considered
                 unknown, and the similarity values appear with a green background whenever it is above 0.5 and 
                 with a red background whenever it is under 0.5.
    """
    # Device
    if 'device' in kwargs.keys():
        device = kwargs['device']
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    # Network
    if type(network) == str: # Load network state dictionary from path
        net = networks.SymbolNet()
        net.load_state_dict(torch.load(network, map_location=device))
    else: # Else we assume network is already loaded with correct state dictionary
        net = network
    
    # Other definitions
    to_im = transforms.ToPILImage()
    num_examples  =  kwargs['num_examples']  if ('num_examples'  in kwargs) else 10
    split          = kwargs['emnist_split']  if ('emnist_split'  in kwargs) else 'balanced'
    classes_string = kwargs['classes']       if ('classes'       in kwargs) else 'digits'
    label          = kwargs['label']         if ('label'         in kwargs) else None
    
    # Image to be classified
    image = image.to(device)
    im_batch = image.repeat(num_examples,1,1,1)

    # Get example images from each class
    classes, class_labels = dataloaders.EmnistClassLabelsSetup(split, classes_string)
    num_classes = len(classes)
    train_ds = dataloaders.emnist(data_path=data_path, split=split, train=True)
    examples = example_assembly(train_ds, class_labels, num_examples, device=device)
    
    # Create image
    image_tensors = []
    for i in range(num_classes):
        if label == class_labels[i]:
            t = torch.ones(num_examples).to(device)
        elif label is not None:
            t = torch.zeros(num_examples)
        else:
            raise ValueError("None label not implemented yet")
        batch = (im_batch, examples[i], t)
        _, image_tensor = visualize_pairs(network, batch, device=device)
        image_tensors.append(image_tensor)
    image_tensors = torch.stack(image_tensors, dim=0)
    buffer = torch.stack([torch.ones((3, image_tensors.shape[2], 10)) * 0.5] * num_classes, dim=0)
    full_tensor = torch.cat((image_tensors, buffer), dim=3)
    full_tensor = utils.make_grid(full_tensor, nrow = num_classes, padding = 0)
    full_tensor = full_tensor[:,:,0:-10] # Remove last buffer
    full_image = to_im(full_tensor)
    
    # Save
    if 'save_path' in kwargs:
        full_image.save(kwargs['save_path'])
    
    # Show image
    if ('show_image' in kwargs) and bool(kwargs['show_image']):
        full_image.show()
    
    return full_image, full_tensor
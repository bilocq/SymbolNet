# SymbolNet
### A simple example of transfer learning using PyTorch and the EMNIST dataset
SymbolNet is a neural network that accomplishes transfer learning in the context of image classification: it can be trained using a first group of image categories and applied to classify images in an second group of categories that may be disjoint from the first. The only requirement is that the idea of what constitutes a category remains similar throughout the two groups. For instance, you could train SymbolNet by showing it only handwritten letters (where the categories are "A", "B", "C", etc) and then use it to classify handwritten digits (where the categories are "1", "2", "3", etc) with a reasonably high accuracy. I've experimented with SymbolNet using the EMNIST dataset, which contains handwritten digits as well as uppercase and lowercase letters, and this codebase is designed for that purpose. However, there's no constraint that would prevent SymbolNet from being used on other types of images. 

The basic idea is that instead of learning to recognize individual categories of images, like a standard classification neural network, **SymbolNet learns to recognize whether or not two images fall into the same category**. In practice, SymbolNet takes as input a pair of images and outputs a number that I call a similarity score: the higher this score is, the more likely SymbolNet thinks it is that the two images are in the same category. Effectively, this means SymbolNet is designed to learn what "*category*" means in a certain context, which gives it much more flexibility than a standard classification network.


### Why is SymbolNet more flexible?
Let's take an example. If you had a vanilla classification neural network trained to recognize handwritten versions of the letters "A" to "J", then that network's outputs would consist of ten numbers each corresponding to one of these letters. If you suddenly decided you wanted to change the network so it could recognize the letter "K" as well, then you would have to change the network's architecture so its output would contain an eleventh number, and you would also need to at least partially retrain the network to make it learn what a handwritten "K" looks like. This is a bit annoying; after all, if you wanted to teach to a person a new symbol they've never seen - say a letter in an alphabet they don't know - it would be enough just to show them one or two examples of that symbol. There's no extra training required for that person to learn the new symbol, only a bit of memorizing! This is the main inspiration for SymbolNet. By keeping a few "memorized" examples of images from various categories (that weren't necessarily used during training), SymbolNet can use its capacity to judge whether two images are in the same category to classify a new image, or to decide the new image is from an unknown category.



### What's included in this repo
* The SymbolNet architecture, under symbolnet_utils/networks.py. The SymbolNet neural network architecture consists of a convolutional "feature network" that is meant to extract relevant features from images, followed by a fully connected "evaluator" network which maps features to an output of two numbers, which correspond to the pair of images being classified as from the same category or from different categories. There are two settings to decide on when creating an instance of SymbolNet:
  - The feature network can be made deeper or shallower, containing between one and three convolutional "blocks". We can choose the outputs of which of these blocks are to be used as inputs to the the evaluator network. In my rather informal experiments, the best results were obtained when the outputs of all three blocks were shown to the evaluator, but training is faster if we include fewer blocks in the feature network. In train.py, this setting corresponds to the "blocks" argument. The default is to include all three blocks.
  - The two images in an input pair are mapped through the feature network separately, and the resulting features are then combined before being shown to the evaluator. This combination can either be a subtraction or a concatenation. The default is a subtraction.

* The main training function for SymbolNet, in train.py. It relies on a data loader which yields pairs of images with a specific proportion of the pairs containing images from the same class.

* Two ways of testing an instance of SymbolNet: 
  - A "pairing test", where pairs of images are shown to the network and it has to decide whether or not the two images belong to the same category.
  - A classification test, where the network compares new images to memorized images from various categories to attempt to classify the new images.

* A code to "register memory" of requested categories, in register_memory.py. The memory of a category consists in a set of features at the output of each of the feature network's convolutional blocks, obtained by mapping images from that category through the feature network. The "num_mem" argument decides how many sets of features to memorize for each category, whereas the "num_see" decides how many images are used to get these sets of features. If num_see > num_mem, then a K-means algorithm is used to produce the memorized features.

* The train_and_test.sh script can be used to train and test an instance of SymbolNet all at once. With the default settings, it will train using uppercase letters and perform pairing and classification tests on both uppercase letters and digits separately. The tests with digits are meant to demonstrate the transfer learning capabilities of SymbolNet. It will perform memory registration to create a memory file used during the classification tests.

* An interactive mode where you can interact with an instance of SymbolNet by drawing your own symbols. You can even make SymbolNet memorize new categories of symbols. 

* Some other visualization tools, in symbolnet_utils/visualize.py

More detailed explanations can be found as comments within the scripts contained in this repo. Note that throughout this repository, the "classes" argument is used to select which categories of symbols from the EMNIST dataset to use. For examples, classes='uppercases' means all the uppercase letters will be used, and classes='digits_lowercases' means the digits and lowercase letters will be used. A detailed explanation of how to specify other sets of symbol categories can be found in the comment in symbolnet_utils/emnist_tools.py in the EmnistClassLabelsSetup function.


### How to run this code
Create a Python virtual environment and install the requirements file (pip install -r requirements.txt). I've tested this code with Python 3.8. The train.py, pairing_test.py, classification_test.py, register_memory.py and interactive.py scripts can all be run from the command line with the appropriate arguments. 

When running train.py, the "folder" arguments (expects a path to a folder) determines whether we start from scratch (empty or non-existent folder) or resume from previous training (folder created by a previous run train.py). Once the first training run is completed, the specified folder should look like the "pretrained_example" folder in this repo, except for the memory.pkl which gets created when running register_memory.py. The "folder" argument in any of the other main scripts should point to a folder that was created by a previous run of train.py. Note that the use of the "check_folder" function, in symbolnet_utils/folder_tool.py, makes sure that pre-existing files and folder won't be overwritten by specifying the wrong value for the "folder" argument.


## Acknowledgments
The training_stats.py script is an adaptation of the plot.py script from https://github.com/caogang/wgan-gp ((c) 2017 Ishaan Gulrajani). Distributed under the MIT licence.

The EMNIST dataset can be found here: https://www.nist.gov/itl/products-and-services/emnist-dataset.

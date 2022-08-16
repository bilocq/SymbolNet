# SymbolNet
### A simple example of transfer learning using PyTorch and the EMNIST dataset
SymbolNet is a neural network that accomplishes transfer learning in the context of image classification: it can be trained using a first group of image categories and applied to classify images in an second group of categories that may be disjoint from the first. The only requirement is that the idea of what constitutes a category remains similar throughout the two groups. For instance, you could train SymbolNet by showing it only handwritten letters (where the categories are "A", "B", "C", etc) and then use it to classify handwritten digits (where the categories are "1", "2", "3", etc) with a reasonably high accuracy. I've experimented with SymbolNet using the EMNIST dataset, which contains handwritten digits as well as uppercase and lowercase letters (hence the name), and this codebase is designed for that purpose. However, there's no hard constraint that would prevent SymbolNet from being used on other types of images.

The basic idea is that instead of learning to recognize individual categories of images, like a standard classification neural network, **SymbolNet learns to recognize whether or not two images fall into the same category**. In practice, SymbolNet takes as input a pair of images and outputs a number that I call a similarity score: the higher this score is, the more likely SymbolNet thinks it is that the two images are in the same category. Effectively, this means SymbolNet is designed to learn what a category is in a certain context, which gives it much more flexibility than a standard classification network.


### Why is SymbolNet more flexible?
Let's take an example. If you had a vanilla network trained to recognize handwritten versions of the letters "A" to "J", then that network's outputs would consist of ten numbers each corresponding to one of these letters. If you suddenly decided you wanted to change the network so it could recognize the letter "K" as well, then you would have to change the network's architecture so its output would contain an eleventh number, and you would also need to at least partially retrain the network to make it learn what a handwritten "K" looks like. This is a bit annoying; after all, if you wanted to teach to a person a new symbol they've never seen - say a letter in an alphabet they don't know - it would be enough just to show them one or two examples of that symbol. There's no extra training required for that person to learn the new symbol, only a bit of memorizing! This is the main inspiration for SymbolNet. By keeping a few "memorized" examples of images from various categories (that weren't necessarily used during training), SymbolNet can use its capacity to judge whether two images are in the same category to classify a new image, or to decide the new image is from an unknown category.



### What's included in this code
Coming soon



### How to run this code
Coming soon

# SymbolNet
### A simple example of transfer learning using the EMNIST dataset

The most vanilla machine learning problem is possibly the recognition of handwritten symbols. SymbolNet is a neural network that accomplishes transfer learning in the context of symbol recognition: it can be trained using a first group of symbols and applied to classify images in an arbitrary second group of symbols that may be disjoint from the first. For instance, you could train SymbolNet by showing it only handwritten capital letters, and then use it to classify handwritten digits with a reasonably high accuracy.

The basic idea is that instead of learning to recognize individual classes of symbols, like a standard classification neural network, SymbolNet learns to recognize whether *a pair of symbols* represent the same class or different classes. Effectively, this means SymbolNet is designed to learn what "*class*" means in a certain context, which gives it much more flexibility than a standard classification network. 

Let's take an example. If you had a vanilla network trained to recognize handwritten versions of the letters "A" to "J", that network's outputs would consist of ten numbers, each corresponding to one of these letters. If you suddenly decided you wanted to change the network so it could recognize the letter "K" as well, then you would have to change the network's architecture so its output would contain an eleventh number, and you would then need to at least partially retrain the network to make it learn what a handwritten "K" looks like. This is a bit annoying; after all, if you wanted to teach a new symbol to a person they've never seen (say a letter in an alphabet they don't know), it would be enough just to show them one or two examples of that symbol! 



### What's included in this code





### How to run this code

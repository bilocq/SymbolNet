# SymbolNet
A simple example of transfer learning using the EMNIST dataset.

The most vanilla machine learning problem is possibly the recognition of hand written symbols. SymbolNet is a neural network that accomplishes transfer learning in the context of symbol recognition: it can be trained using a first group of symbols and applied to classify images in an arbitrary second group of symbols that may be disjoint from the first. For instance, you could train SymbolNet by showing it only handwritten capital letters, and then use it to classify handwritten digits with a reasonably high accuracy.

The basic idea is that instead of learning to recognize individual classes of symbols, SymbolNet learns to recognize when *a pair of symbols* represent the same class or different classes.  

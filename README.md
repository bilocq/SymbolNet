# SymbolNet
### A simple example of transfer learning using the EMNIST dataset

SymbolNet is a neural network that accomplishes transfer learning in the context of image classification: it can be trained using a first group of image classes and applied to classify images in an second group of classes that may be disjoint from the first. The only requirement is that the idea of what constitutes a class remains similar throughout the two groups. For instance, you could train SymbolNet by showing it only handwritten letters, and then use it to classify handwritten digits with a reasonably high accuracy. 

The basic idea is that instead of learning to recognize individual classes of symbols, like a standard classification neural network, SymbolNet learns to recognize whether *two symbols* represent the same class or different classes. Effectively, this means SymbolNet is designed to learn what a "*class*" is in a certain context, which gives it much more flexibility than a standard classification network. 

Let's take an example. If you had a vanilla network trained to recognize handwritten versions of the letters "A" to "J", then that network's outputs would consist of ten numbers each corresponding to one of these letters. If you suddenly decided you wanted to change the network so it could recognize the letter "K" as well, then you would have to change the network's architecture so its output would contain an eleventh number, and you would also need to at least partially retrain the network to make it learn what a handwritten "K" looks like. This is a bit annoying; after all, if you wanted to teach a new symbol they've never seen to a person (say a letter in an alphabet they don't know), it would be enough just to show them one or two examples of that symbol. There's no extra training required for that person to learn the new symbol, only a bit of memorizing! 



### What's included in this code





### How to run this code

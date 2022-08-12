"""
Used to keep track of statistics during training.
Adapted from github.com/caogang/wgan-gp
"""

import os
import numpy as np
import collections
import pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_iter = [1]

def load(folder):
    with open(os.path.join(folder, 'training_stats.pkl'), 'rb') as file:
        old_stats = pickle.load(file)
        for stat in old_stats:
            _since_beginning[stat] = old_stats[stat]
    
def tick(n=1):
    _iter[0] += n

def offset(n):
    _iter[0] += n

def plot(name, value):
    _since_last_flush[name][_iter[0]] = value

def flush(folder, verbose=False):
    prints = []
    
    for name, vals in _since_last_flush.items():
        prints.append("{:.{prec1}}\t{:.{prec2}f}".format(name, np.mean(list(vals.values())),prec1 = 5, prec2 = 3))
        _since_beginning[name].update(vals)

    if verbose:
        print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
        
    _since_last_flush.clear()

    with open(os.path.join(folder, 'training_stats.pkl'), 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
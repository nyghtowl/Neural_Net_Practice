'''
Code to load the sample MNIST file
'''
import cPickle, gzip
import numpy as np

def load_file(filename):
    with gzip.open(filename) as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return train_set, valid_set, test_set

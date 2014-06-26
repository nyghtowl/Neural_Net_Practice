'''
Code to load the sample MNIST file
'''
from DeepLearning.python import DBN
import cPickle, gzip
import numpy as np
import Image
import random
import time

def load_file(filename):
    with gzip.open(filename) as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return train_set, valid_set, test_set

def show_img(x):
    im = Image.new('L', (28, 28))
    im.putdata(x, scale=256)
    im.show()

def pull_sample(data, size, target):
    # x is pic pixels and round up or down
    # y is numeric labels
    # n sets size of sample
    x = data[0][data[1]==target][0:size].round()
    y = data[1][data[1]==target][0:size]
    return x, y

# def binarize(num):
#     # need to convert y labels
#     if num == 0:
#         return [1,0]
#     else:
#         return [0,1]

def build_sample(data):
    x_results, y_results = [], []
    for value in xrange(0,2):
        x, y = pull_sample(data, 10, value)
        x_results.append(x)
        y_results.append(y)

    x_sample = np.vstack(x_results)
    y_sample = np.concatenate(y_results)

    result = zip(x_sample, y_sample)
    random.shuffle(result)
    return zip(*result)


def main():
    train_set, valid_set, test_set = load_file('../mnist.pkl.gz')
    #show_img(train_set[0][100]) # see example of image
    #print train_set[1][100] # confirm label associated
    pics, labels = build_sample(train_set)
    return DBN.DBN(input=pics, label=labels, n_ins=784, hidden_layer_sizes=[500, 250, 100], n_outs=10, numpy_rng=None)



if __name__ == "__main__":
    main()
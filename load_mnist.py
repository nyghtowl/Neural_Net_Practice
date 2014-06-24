'''
Code to load the sample MNIST file
'''
import cPickle, gzip
import numpy as np
import Image

def load_file(filename):
    with gzip.open(filename) as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return train_set, valid_set, test_set

def show_img(x):
    im = Image.new('L', (28, 28))
    im.putdata(x, scale=256)
    im.show()

def build_sample(data, size, target):
    # x is pic pixels and round up or down
    # y is numeric labels
    # n sets size of sample
    x = data[0][data[1]==target][0:size].round()
    y = data[1][data[1]==target][0:size]
    return x, y

def main():
    train_set, valid_set, test_set = load_file('../mnist.pkl.gz')
    show_img(train_set[0][100]) # see example of image
    print train_set[1][100] # confirm label associated
    x, y = build_sample(train_set, 10, 0)

if __name__ == "__main__":
    main()
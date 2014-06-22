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

# Not working - possibly need to reformat data into numpy...
def show_img(x):
    im = Image.new('L', (28, 28))
    im.putdata(x, scale=256)
    im.show()

def main():
    train_set, valid_set, test_set = load_file('../mnist.pkl.gz')
    show_img(train_set)

if __name__ == "__main__":
    main()
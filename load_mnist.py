
'''
Code to load the sample MNIST file
'''
from DeepLearning.python import DBN
import cPickle, gzip
import numpy as np
import Image
import random
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
from pylab import cm
import seaborn

def load_file(filename):
    with gzip.open(filename) as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return train_set, valid_set, test_set

###########################################################

# Explore & Split Data

def eda(df):
    print "Summary Stats", df.describe()
    print "Shape", df.shape

    print "# email threads", len(df.thread_id.unique())
    print "# email threads that meet conditions", len(df[df.target == True].thread_id.unique())

    print "Top 5 rows", df.head()
    print "Bottom 5 rows", df.tail()

def show_img(x):
    im = Image.new('L', (28, 28))
    im.putdata(x, scale=256)
    im.show()

def split_data(data, size, target):
    # x is pic pixels and round up or down
    # y is numeric labels
    # n sets size of sample
    x = data[0][data[1]==target][0:size].round()
    y = data[1][data[1]==target][0:size]
    return x, y

# Alternative to adjust it when have more than 2 labels
def binarize_label(num):
    label = [0] * 10
    label[num] = 1
    return label

def create_data_sample(data, size):
    x_results, y_results = [], []
    for value in xrange(5,8,2):
        x, y = split_data(data, size, value)
        x_results.append(x)
    
        y_results.append([binarize_label(num) for num in y.tolist()])

    x_sample = np.vstack(x_results)
    y_sample = np.vstack(y_results)

    result = zip(x_sample, y_sample)
    random.shuffle(result)
    return zip(*result)

###########################################################

# Model

def build_model(pics, labels, lr=0.0035, epochs=5000):
    labels = np.array(labels)
    pics = np.array(pics)

    # lr is learning rate - start with .001
    # epochs is ht enumber iterations to run - start at 1000

    model = DBN.DBN(input=pics, label=labels, n_ins=784, hidden_layer_sizes=[500, 250, 100], n_outs=10, numpy_rng=None)
    model.pretrain(lr=lr, epochs=epochs) # feature extraction

    # build/fit model by running logistic regression
    model.finetune(lr=.001, epochs=epochs) 

    return model

###########################################################

# Analyze Results

def print_accuracy(model, labels, values):
    print sum(np.argmax(labels, axis=1) == np.argmax(model.predict(values), axis=1))*1.0/len(labels)

def create_confusion_matrix(y_test, y_pred, cmap=cm.cubehelix_r):
    # Change cm_lables to receive input
    # cm_labels = [True, False]
    conf_matrix = confusion_matrix(y_test, y_pred)
    print 'Neural Net CM:'
    print conf_matrix
    print
    cm_plot = plot_confusion_matrix(conf_matrix, cmap)
    return conf_matrix

def plot_confusion_matrix(conf_matrix, cmap):

    startcolor = '#cccccc'
    midcolor = '#08519c'
    endcolor = '#08306b'

    b_g2 = LinearSegmentedColormap.from_list('B_G2', [startcolor, midcolor, endcolor])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix, cmap=b_g2)
    fig.colorbar(cax)
    plt.title('Neural Net Confusion Matrix \n', fontsize=16)

    # ax.set_xticklabels([''] + cm_labels, fontsize=13)
    # ax.set_yticklabels([''] + cm_labels, fontsize=13)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    spines_to_remove = ['top', 'right', 'left', 'bottom']

    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    #plt.savefig(os.path.join(graph_dir, graph_fn))

    plt.show()


def main():
    train_set, valid_set, test_set = load_file('../mnist.pkl.gz') # outputs tuples
    #show_img(train_set[0][100]) # see example of image
    #print train_set[1][100] # confirm label associated

    train_pics, train_labels = create_data_sample(train_set, 50)

    start = time.time()
    dbn = build_model(train_pics, train_labels)
    print "Time to train model:", time.time() - start

    return dbn


if __name__ == "__main__":
    main()
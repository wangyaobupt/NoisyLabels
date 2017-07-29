import tensorflow as tf
import numpy as np
from simpleCNN import SimpleCNN
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

'''
Test case 2: use random noise on training set, no noise on validation set, evaluating the degrade of model of different noise level
'''

# Add random noise to MNIST training set
# input:
#       mnist_data: data structure that follow tensorflow MNIST demo
#       noise_level: a percentage from 0 to 1, indicate how many percentage of labels are wrong
def addRandomNoiseToTrainingSet(mnist_data, noise_level):
    # the data structure of labels refer to DataSet in tensorflow/tensorflow/contrib/learn/python/learn/datasets/mnist.py 
    label_data_set = mnist_data.train.labels
    #print label_data_set.shape

    totalNum = label_data_set.shape[0]
    corruptedIdxList = randomSelectKFromN(int(noise_level*totalNum),totalNum)
    #print 'DEBUG: 1st elements in corruptedIdxList is: ', corruptedIdxList[0], ' length = ', len(corruptedIdxList)

    for cIdx in corruptedIdxList:
        #print "DEBUG: convert index = ", cIdx
        correctLabel = label_data_set[cIdx]
        #print 'DEBUG: Correct label = ', correctLabel
        wrongLabel = convertCorrectLabelToCorruptedLabel(correctLabel)
        #print 'DEBUG: Wrong label = ', wrongLabel
        label_data_set[cIdx] = wrongLabel


# uniform randomly select K integers from range [0,N-1]
def randomSelectKFromN(K, N):
    #print 'DEBUG: K = ',K, ' N = ', N
    resultList =[]
    seqList = range(N)
    while (len(resultList) < K):
        index = (int)(np.random.rand(1)[0] * len(seqList))
        #index = 0 # for DEBUG ONLY
        resultList.append(seqList[index])
        seqList.remove(seqList[index])
    #print resultList
    return resultList

# Convert correct ont-hot vector label to a wrong label, the error pattern is randomly selected, i.e. not considering the content of image
def convertCorrectLabelToCorruptedLabel(correctLabel):
    correct_value = np.argmax(correctLabel, 0)
    target_value = int(np.random.rand(1)[0]*10)%10
    if target_value == correct_value:
        target_value = ((target_value+1) % 10)
    result = np.zeros(correctLabel.shape)
    result[target_value] = 1.0 
    return result

if __name__ == '__main__':
    mnist = tf.examples.tutorials.mnist.input_data.read_data_sets('MNIST_DATA/', one_hot=True)
    #print 'DEBUG: in main: before add noise', mnist.train.labels[0]
    addRandomNoiseToTrainingSet(mnist,0.01)
    #print 'DEBUG: in main: after add noise', mnist.train.labels[0]
    cnn = SimpleCNN(1e-4)
    cnn.train(mnist)
    cnn.test(mnist)

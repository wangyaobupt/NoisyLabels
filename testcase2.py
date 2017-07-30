import tensorflow as tf
import numpy as np
from simpleCNN import SimpleCNN
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from datetime import datetime
'''
Test case 2: use random noise (i.e. white noise) on training set, no noise on validation set, evaluating the degrade of model of different noise level
the noise is 'white', meaning the error pattern is uniformly distributed. For example, the correct label for an image is '1', when noise is applied, it has equal opportunity to be changed to 0,2,3,4,5,6,7,8,9
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

def testOnCertainNoiseLevel(noiseLevel):
    print '', datetime.now().isoformat(), 'noise = ', noiseLevel
    mnist = tf.examples.tutorials.mnist.input_data.read_data_sets('MNIST_DATA/', one_hot=True)
    addRandomNoiseToTrainingSet(mnist, noiseLevel)
    cnn = SimpleCNN(1e-4)
    cnn.train(mnist)
    result = cnn.test(mnist)
    del cnn
    return result

if __name__ == '__main__':
    noiseList = np.linspace(0,1,100)
    result = {}
    for noise in noiseList:
        result[str(noise)] = testOnCertainNoiseLevel(noise)
    
    for noise in noiseList:
	print '',noise,', ',result[str(noise)]

import tensorflow as tf
import numpy as np
from simpleCNN import SimpleCNN

'''
This is the basic test case which user 100% correct labels to train and evaluate models
'''
if __name__ == '__main__':
    mnist = tf.examples.tutorials.mnist.input_data.read_data_sets('MNIST_DATA/', one_hot=True)
    cnn = SimpleCNN(1e-4)
    cnn.train(mnist)
    cnn.test(mnist);

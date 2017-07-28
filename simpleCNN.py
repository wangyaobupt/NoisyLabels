import tensorflow as tf
from mnist_deep import deepnn

'''
SimpleCNN is a wrapper class of MNIST CNN demo code
'''
class SimpleCNN:
    def __init__(self, lr):
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        y_conv, self.keep_prob = deepnn(self.x)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, mnist):
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = self.accuracy.eval(session=self.sess, feed_dict={
                    self.x: batch[0], self.y_: batch[1],self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            self.train_step.run(session=self.sess, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

    def test(self, mnist):
        print('test accuracy %g' % self.accuracy.eval(session=self.sess, feed_dict={
            self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))



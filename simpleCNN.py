import tensorflow as tf
import os
import os.path
from mnist_deep import deepnn
from datetime import datetime
import numpy as np

'''
SimpleCNN is a wrapper class of MNIST CNN demo code
'''
class SimpleCNN:
    def __init__(self, lr, log_path='tf_writer/', max_output_images=16):
        # clean log_path
        removeFileInDir(log_path)

        # Create the model
        
        self.x = tf.placeholder(tf.float32, [None, 784])
        # visualize X
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        tf.summary.image('input_image', x_image, max_outputs=max_output_images)
        
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        # visualize Y
        label_str = tf.as_string(self.y_)
        tf.summary.text('label', label_str)

        # Build the graph for the deep net
        y_conv, self.keep_prob = deepnn(self.x)
        self.output_prob_distribution = tf.nn.softmax(y_conv, name='out_prob_dist')
        

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            self.label = tf.argmax(self.y_, 1)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), self.label)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)
            tf.summary.scalar('accuracy',self.accuracy)

        self.merged = tf.summary.merge_all()

        all_vars = tf.global_variables()
        self.saver = tf.train.Saver(all_vars)

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(log_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
   
    def reset(self):
        self.sess.run(tf.global_variables_initializer())
        print 'Re-Init the graph parameter'
 
    def train(self, mnist):
        for i in range(1100):
            batch = mnist.train.next_batch(5000)
            if i % 100 == 0:
                # Console debug output
                train_accuracy, prob_dist, label=self.sess.run([self.accuracy, self.output_prob_distribution, self.label], feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob:1.0})
                print(datetime.now().isoformat(), 'step %d, training accuracy %g' % (i, train_accuracy))
                for sample_idx in range(16):
                    sample_str = 'sample_idx=%d, Output_Probability_Distribution = [' % sample_idx
                    for class_idx in range(prob_dist.shape[1]):
                        sample_str += ("%.3f," % (prob_dist[sample_idx][class_idx]))
                    sample_str += ("] ")
                    sample_str += ("Label = %d" % label[sample_idx])
                    print sample_str
                
                # Tensorboard debug output
                merged = self.sess.run(self.merged, feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob:1.0})
                self.writer.add_summary(merged, i)
                self.writer.flush()
                
            self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

    # Filter out images with low SNR.
    # The term 'low SNR' is defined as: in the probability distribution of this sample, the largest value is <= 0.7, while the 2nd largest value >= 0.15
    # the raw images data (in shape of 1*784 vector), labels, and top 2 possibilities by CNN will be returned
    # Parameter:
    # train_or_test, 0 means train data, 1 means test data
    def filterLowSNRSamples(self, mnist, train_or_test=0):
        if train_or_test == 1:
            data = mnist.test
        else:
            data = mnist.train
        
        resultList = []

        for sample_idx in range(data.images.shape[0]):
            prob_dist, label=self.sess.run([self.output_prob_distribution, self.label], feed_dict={
                    self.x: np.reshape(data.images[sample_idx], (1, 784)), self.y_: np.reshape(data.labels[sample_idx], (1,10)), self.keep_prob:1.0})
            # search for position of the largest value and the 2nd largest value
            raw_prob_array = prob_dist[0]
            top_1_pos = 0
            top_2_pos = 0
            for j in range(1, 10):
                if (raw_prob_array[top_1_pos] < raw_prob_array[j]):
                    top_1_pos = j

                if j != top_1_pos and (raw_prob_array[top_2_pos] < raw_prob_array[j]):
                    top_2_pos = j
            
            #Low SNR criteria
            if raw_prob_array[top_1_pos] <= 0.7 and raw_prob_array[top_2_pos] >= 0.15:
                resultList.append((sample_idx, data.images[sample_idx], label, top_1_pos, top_2_pos))

            if (sample_idx % 1000 == 0):
                print "DEBUG, current idx = %d, num_of_low_SNR = %d" % (sample_idx, len(resultList))
        return resultList

    def test(self, mnist):
        acc_result = self.accuracy.eval(session=self.sess, feed_dict={
            self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})
        print(datetime.now().isoformat(), 'test accuracy %g' % acc_result)
        return acc_result

def removeFileInDir(targetDir): 
    for file in os.listdir(targetDir): 
        targetFile = os.path.join(targetDir,  file) 
        if os.path.isfile(targetFile):
            print 'Delete Old Log FIle:', targetFile
            os.remove(targetFile)


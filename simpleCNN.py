import tensorflow as tf
import os
import os.path
from mnist_deep import deepnn
from datetime import datetime

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
        tf.summary.image('input_image', x_image,max_outputs=max_output_images)
        
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        # visualize Y
        label_str = tf.as_string(self.y_)
        tf.summary.text('label', label_str)

        # Build the graph for the deep net
        y_conv, self.keep_prob, self.w_conv1, self.w_conv2  = deepnn(self.x)
        self.visualizeKernelToTensorboard('1st_layer_kernel_image', self.w_conv1)
        #self.visualizeKernelToTensorboard('2nd_layer_kernel_image', self.w_conv2)
        
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)
            tf.summary.scalar('accuracy',self.accuracy)

        self.merged = tf.summary.merge_all()

        all_vars = tf.global_variables()
        self.saver = tf.train.Saver(all_vars)

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(log_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
    
    def visualizeKernelToTensorboard(self, name_of_image, w_conv):
        num_kernels = int(w_conv.shape[3])
        height = int(w_conv.shape[0])
        width = int(w_conv.shape[1])
        channel = int(w_conv.shape[2])
        slice_list = []
        for k_idx in range(num_kernels):
            each_slice = tf.slice(w_conv,[0,0,0,k_idx],[height,width,channel,1])
            reshaped_slice = tf.reshape(each_slice, [height, width, channel])
            slice_list.append(reshaped_slice)
        kernel_image = tf.stack(slice_list)
        tf.summary.image(name_of_image, kernel_image, max_outputs=num_kernels)

    def reset(self):
        self.sess.run(tf.global_variables_initializer())
        print 'Re-Init the graph parameter'
 
    def train(self, mnist):
        for i in range(2000):
            batch = mnist.train.next_batch(1000)
            if i % 100 == 0:
                train_accuracy = self.accuracy.eval(session=self.sess, feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob:1.0})
                print(datetime.now().isoformat(), 'step %d, training accuracy %g' % (i, train_accuracy))
                merged = \
                    self.sess.run(self.merged, feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob:1.0})

                self.writer.add_summary(merged, i)
                self.writer.flush()
            self.train_step.run(session=self.sess, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

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


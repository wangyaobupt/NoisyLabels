import tensorflow as tf
import numpy as np
from simpleCNN import SimpleCNN
from PIL import Image

'''
Filter out low SNR samples
'''
if __name__ == '__main__':
    mnist = tf.examples.tutorials.mnist.input_data.read_data_sets('MNIST_DATA/', one_hot=True)
    cnn = SimpleCNN(1e-4)
    cnn.train(mnist)
    resultList = cnn.filterLowSNRSamples(mnist);

    col=28
    row=28
    with open("lowSNR.csv","w") as f:
        f.write("Seq id, Label,top_1_class, top_2_class\n")
        for sampleIdx in range(len(resultList)):
            f.write('%d, %d,%d,%d\r\n' % (resultList[sampleIdx][0], resultList[sampleIdx][2], resultList[sampleIdx][3], resultList[sampleIdx][4]))
            image_data = np.reshape(resultList[sampleIdx][1], (col, row))
            output_filename = "seq_%d_label_%d_top_class_%d_second_class_%d.png" % (resultList[sampleIdx][0],resultList[sampleIdx][2], resultList[sampleIdx][3], resultList[sampleIdx][4])
            image_data=image_data*255
            im = Image.fromarray(image_data).convert('L')
            im.save(output_filename)



import tensorflow as tf
import tensorflow.contrib.layers.python.layers as tf_new
import numpy as np
import pdb

class AudioCNN(object):
    """
    A CNN for coversong identification. Pipelines based on FCN-4 from Choi et. al. 2016 https://arxiv.org/abs/1606.00298 
    Uses two pipelines of four convolutional layers & max-pooling layers followed tied together by a binary softmax layer.
    """
    def __init__(
      self, spect_dim, num_classes, gap_reg = 0.0001,
      filters_per_layer, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.query = tf.placeholder(tf.float32, [None, *spect_dim], name="query")
        self.similar = tf.placeholder(tf.float32, [None, *spect_dim], name="similar")
        self.different = tf.placeholder(tf.float32, [None, *spect_dim], name="different")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # create wrappers for basic convnet functions
        def create_variable(self, name, shape, initializer = tf.random_normal_initializer()):
            '''
            Function wrapper for creating a layer with random initializer
            '''
            return tf.get_variable(name, shape, initializer=initializer)

        def conv(self, x, kx, ky, in_depth, num_filters, sx=1, sy=1, name=None, reuse=None, batch_norm=True):
            '''
            Function that defines a convolutional layer
            -------------------------------------------
            x           : input tensor
            kx,ky       : filter (kernel) dimensions
            sx,sy       : stride dimensions
            in_depth    : depth of input tensor
            num_filters : number of conv filters
            '''
            with tf.variable_scope(name, reuse=reuse) as scope:
                kernel = create_variable(self, "weights", [kx, ky, in_depth, num_filters], tf.contrib.layers.xavier_initializer_conv2d())
                bias = create_variable(self, "bias", [num_filters])
                conv = tf.nn.relu(tf.nn.bias_add(
                       tf.nn.conv2d(x, kernel, strides=[1, sx, sy, 1], padding='SAME'), bias),
                                    name=scope.name)
                if batch_norm:
                    # batch normalization
                    conv = tf_new.batch_norm(conv, scale=False)
            return conv

        def pool(self, x, kx, ky, sx=None, sy=None, name=None):
            '''
            Function that defines a pooling layer
            If no specified stride: stride = kernel size
            -------------------------------------------
            x           : input tensor
            kx,ky       : kernel dimensions
            sx,sy       : stride dimensions
            '''            
            if not sx or sy: sx,sy = kx,ky
            pool = tf.nn.max_pool(x, ksize=[1, kx, ky, 1], strides=[1, sx, sy, 1], padding='SAME')
            return pool

        def euclidean_distance(self, x, y, name=None):
            '''
            Function that gives the euclidean distance
            between two tensors
            -------------------------------------------
            x           : input tensor
            y           : input tensor
            '''
            with tf.variable_scope(name) as scope:
                dist = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x, y)), reduction_indices=1))
            return dist

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        #def conv_architecture(song,name_scope):
        with tf.name_scope("conv-query"), tf.device('/gpu:0'):
            # convolutional architecture for first song ('original song')
            conv1a = conv(self, x=tf.expand_dims(self.query,-1), kx=3, ky=3, in_depth=1, num_filters=filters_per_layer[0], name='conv1a')
            conv1a = pool(self, conv1a, kx=2, ky=4, name='pool1a')
            # conv2a
            conv2a = conv(self, x=conv1a, kx=3, ky=3, in_depth=filters_per_layer[0], num_filters=filters_per_layer[1], name='conv2a')
            conv2a = pool(self, conv2a, kx=3, ky=5, name='pool2a')
            # conv3a
            conv3a = conv(self, x=conv2a, kx=3, ky=3, in_depth=filters_per_layer[1], num_filters=filters_per_layer[2], name='conv3a')
            conv3a = pool(self, conv3a, kx=3, ky=8, name='pool3a')
            # conv4a
            conv4a = conv(self, x=conv3a, kx=3, ky=3, in_depth=filters_per_layer[2], num_filters=filters_per_layer[3], name='conv4a')
            conv4a = pool(self, conv4a, kx=5, ky=8, name='pool4a') # 5,8 for 30 sec; 5,17 for 1min
            self.query_out = tf.reshape(conv4a, [-1, filters_per_layer[3]])

        with tf.name_scope("conv-similar"), tf.device('/gpu:1'):
            # convolution architecture for second song ('cover song')
            conv1b = conv(self, x=tf.expand_dims(self.similar,-1), kx=3, ky=3, in_depth=1, num_filters=filters_per_layer[0], name='conv1b')
            conv1b = pool(self, conv1b, kx=2, ky=4, name='pool1b')
            # conv2b
            conv2b = conv(self, x=conv1b, kx=3, ky=3, in_depth=filters_per_layer[0], num_filters=filters_per_layer[1], name='conv2b')
            conv2b = pool(self, conv2b, kx=3, ky=5, name='pool2b')
            # conv3b
            conv3b = conv(self, x=conv2b, kx=3, ky=3, in_depth=filters_per_layer[1], num_filters=filters_per_layer[2], name='conv3b')
            conv3b = pool(self, conv3b, kx=3, ky=8, name='pool3b')
            # conv4b
            conv4b = conv(self, x=conv3b, kx=3, ky=3, in_depth=filters_per_layer[2], num_filters=filters_per_layer[3], name='conv4b')
            conv4b = pool(self, conv4b, kx=5, ky=8, name='pool4b') # 5,8 for 30 sec; 5,17 for 1min       
            self.similar_out = tf.reshape(conv4b, [-1, filters_per_layer[3]])

        with tf.name_scope("conv-different"), tf.device('/gpu:2'):
            # convolution architecture for second song ('cover song')
            conv1c = conv(self, x=tf.expand_dims(self.different,-1), kx=3, ky=3, in_depth=1, num_filters=filters_per_layer[0], name='conv1c')
            conv1c = pool(self, conv1b, kx=2, ky=4, name='pool1c')
            # conv2c
            conv2c = conv(self, x=conv1b, kx=3, ky=3, in_depth=filters_per_layer[0], num_filters=filters_per_layer[1], name='conv2c')
            conv2c = pool(self, conv2b, kx=3, ky=5, name='pool2c')
            # conv3c
            conv3c = conv(self, x=conv2b, kx=3, ky=3, in_depth=filters_per_layer[1], num_filters=filters_per_layer[2], name='conv3c')
            conv3c = pool(self, conv3b, kx=3, ky=8, name='pool3c')
            # conv4c
            conv4c = conv(self, x=conv3b, kx=3, ky=3, in_depth=filters_per_layer[2], num_filters=filters_per_layer[3], name='conv4c')
            conv4c = pool(self, conv4b, kx=5, ky=8, name='pool4c') # 5,8 for 30 sec; 5,17 for 1min       
            self.different_out = tf.reshape(conv4c, [-1, filters_per_layer[3]])

        # Calculate euclidean distances
        self.distance_similar = euclidean_distance(self, self.query_out, self.similar_out, name='distance_similar')
        self.distance_different = euclidean_distance(self, self.query_out, self.different_out, name='distance_different')
        self.predictions = tf.argmax(self.distance_different,self.distance_similar,1)

        # Calculate hinge loss
        with tf.name_scope("loss"), tf.device('/gpu:3'):
            self.loss = tf.max(0.0, tf.sum(gap_reg, tf.sub(self.distance_similar, self.distance_different)))

        # Accuracy
        with tf.name_scope("accuracy"), tf.device('/gpu:3'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
              

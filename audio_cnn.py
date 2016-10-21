import tensorflow as tf
import numpy as np
import pdb

class AudioCNN(object):
    """
    A CNN for coversong identification. Pipelines based on FCN-4 from Choi et. al. 2016 https://arxiv.org/abs/1606.00298 
    Uses two pipelines of four convolutional layers & max-pooling layers followed by one softmax layer.
    """
    def __init__(
      self, spect_dim, num_classes,
      fc_layers=1, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_song1 = tf.placeholder(tf.float32, [None, *spect_dim], name="input_song1")
        self.input_song2 = tf.placeholder(tf.float32, [None, *spect_dim], name="input_song2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # create wrappers for basic convnet functions
        def create_variable(self, name, shape):
            '''
            Function wrapper for creating a layer with random initializer
            '''
            return tf.get_variable(name, shape, initializer=tf.random_normal_initializer())

        def conv(self, x, kx, ky, in_depth, num_filters, sx=1, sy=1, name=None):
            '''
            Function that defines a convolutional layer
            -------------------------------------------
            x           : input tensor
            kx,ky       : filter (kernel) dimensions
            sx,sy       : stride dimensions
            in_depth    : depth of input tensor
            num_filters : number of conv filters
            '''
            with tf.variable_scope(name) as scope:
                kernel = create_variable(self, "weights", [kx, ky, in_depth, num_filters])
                bias = create_variable(self, "bias", [num_filters])
                conv = tf.nn.relu(tf.nn.bias_add(
                       tf.nn.conv2d(x, kernel, strides=[1, sx, sy, 1], padding='SAME'), bias),
                                    name=scope.name)
                #self.add_layer(name, conv)
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

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        #def conv_architecture(song,name_scope):
        with tf.name_scope("conv-song1"):
            # convolutional architecture for first song ('original song')
            conv1a = conv(self, x=tf.expand_dims(self.input_song1,-1), kx=3, ky=3, in_depth=1, num_filters=128, name='conv1a')
            conv1a = pool(self, conv1a, kx=2, ky=4, name='pool1a')
            # conv2a
            conv2a = conv(self, x=conv1a, kx=3, ky=3, in_depth=128, num_filters=384, name='conv2a')
            conv2a = pool(self, conv2a, kx=3, ky=5, name='pool2a')
            # conv3a
            conv3a = conv(self, x=conv2a, kx=3, ky=3, in_depth=384, num_filters=768, name='conv3a')
            conv3a = pool(self, conv3a, kx=3, ky=8, name='pool3a')
            # conv4a
            conv4a = conv(self, x=conv3a, kx=3, ky=3, in_depth=768, num_filters=2048, name='conv4a')
            conv4a = pool(self, conv4a, kx=5, ky=17, name='pool4a')
            self.song1_out = tf.reshape(conv4a, [-1, 2048])

        with tf.name_scope("conv-song2"):
            # convolution architecture for second song ('cover song')
            conv1b = conv(self, x=tf.expand_dims(self.input_song2,-1), kx=3, ky=3, in_depth=1, num_filters=128, name='conv1b')
            conv1b = pool(self, conv1b, kx=2, ky=4, name='pool1b')
            # conv2b
            conv2b = conv(self, x=conv1b, kx=3, ky=3, in_depth=128, num_filters=384, name='conv2b')
            conv2b = pool(self, conv2b, kx=3, ky=5, name='pool2b')
            # conv3b
            conv3b = conv(self, x=conv2b, kx=3, ky=3, in_depth=384, num_filters=768, name='conv3b')
            conv3b = pool(self, conv3b, kx=3, ky=8, name='pool3b')
            # conv4b
            conv4b = conv(self, x=conv3b, kx=3, ky=3, in_depth=768, num_filters=2048, name='conv4b')
            conv4b = pool(self, conv4b, kx=5, ky=17, name='pool4b')            
            self.song2_out = tf.reshape(conv4b, [-1, 2048])

        # concatenate transformed song vectors
        self.songs_vector = tf.concat(1, [self.song1_out,self.song2_out])

        # Add dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        drop = tf.nn.dropout(self.songs_vector, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[4096, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), "b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
              

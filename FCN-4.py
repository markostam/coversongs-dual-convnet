import tensorflow as tf
import numpy as np

class AudioCNN(object):
    """
    A CNN for coversong identification.
    Uses two pipelines of four convolutional layers & max-pooling layers followed by one softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, activation_func,
      fc_layers=1, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # create wrappers for basic convnet functions
        def create_variable(self, name, shape):
            '''
            Function wrapper for creating a layer with random initializer
            '''
            return tf.get_variable(name, shape, initializer=tf.random_normal_initializer())

        def conv(self, kx, ky, sx, sy, input, in_depth, num_filters, name=None):
            '''
            Function that defines a convolutional layer
            -------------------------------------------
            kx,ky       : kernel dimensions
            sx,sy       : stride dimensions
            input       : input tensor
            in_depth    : depth of input tensor
            num_filters : number of conv filters
            '''
            with tf.variable_scope(name) as scope:
                kernel = self.create_variable("weights", [kx, ky, in_depth, num_filters])
                bias = self.create_variable("bias", [num_filters])
                conv = tf.nn.relu(tf.nn.bias_add(
                       tf.nn.conv2d(input, kernel, strides=[1, sx, sy, 1], padding='SAME'), bias),
                                    name=scope.name)
                self.add_layer(name, conv)
            return self

        def pool(self, kx, ky, sx=1, sy=1, name=None):
            name = name or self.get_unique_name("pool")
            input = self.get_last_output()
            pool = tf.nn.max_pool(input, ksize=[1, kx, ky, 1], strides=[1, sx, sy, 1], padding='SAME')
            self.add_layer(name, pool)
            return self

        def fc(self, out_size, name=None):
            name = name or self.get_unique_name("fc")
            with tf.variable_scope(name) as scope:
                input = self.get_last_output()
                shape = input.get_shape().as_list()
                in_size = np.prod(shape[1:])
                weights = self.create_variable("weights", [in_size, out_size])
                bias = self.create_variable("bias", [out_size])
                input_flat = tf.reshape(input, [-1, in_size])
                fc = tf.nn.relu(tf.nn.xw_plus_b(input_flat, weights, bias, name=scope.name))
                self.add_layer(name, fc)
            return self



        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        '''
        # Embedding layer
        with tf.device('/gpu:4'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        '''

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []

        #for i, filter_size in enumerate(filter_sizes):
        filter_size = [3,3]
        num_filters = 128
        depth = 
        with tf.name_scope("conv-maxpool-1"):
            # Convolution Layer
            filter_shape = [filter_size[0], embedding_size[1], 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            if activation_func == 'tanh':
                h = tf.nn.tanh(tf.nn.bias_add(conv, b), name = 'tanh')
            else:
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # check if we have one or two fully connected layers
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
              

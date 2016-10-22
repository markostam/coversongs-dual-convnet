#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import gzip
import pickle
import random
import pdb
from stat_collector import StatisticsCollector
from audio_cnn import AudioCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
#tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
#tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("dropout_factor", 1.0, "Probability of weights to keep for dropout (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", .0005, "Gradient descent learning rate (default: .0005)")
#tf.flags.DEFINE_float("fc_layers", 1, "number of fully connected layers at output (1 or 2) (default: 1)")
#tf.flags.DEFINE_string("activation_func", 'relu', "activation function (can be: tanh or relu) (default: relu)")
tf.flags.DEFINE_float("l2_constraint", None, "Constraint on l2 norms of weight vectors (default: None)")
tf.flags.DEFINE_float("dev_size_percent", 0.10, "size of the dev batch in percent vs entire train set (default: 0.10)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 400, "Evaluate model on dev set after this many steps (default: 400)")
tf.flags.DEFINE_integer("checkpoint_every", 800, "Save model after this many steps (default: 200)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
flags_list = ["{}={}".format(attr.upper(), value) for attr,value in sorted(FLAGS.__flags.items())]
save_flags = ["{}={}".format(attr.upper(), value) for attr,value in sorted(FLAGS.__flags.items())]
for i in flags_list:
    print(i)
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
train_loc = "./shs/shs_dataset_train.txt"
path_to_pickles = "./shs/shs_train_pickles"
spect_dict = data_helpers.read_from_pickles(path_to_pickles)
cliques = data_helpers.txt_to_cliques(train_loc)
pruned_cliques = data_helpers.prune_cliques(cliques,spect_dict)
x, y = data_helpers.get_labels(pruned_cliques)

# Randomly shuffle data
np.random.seed(420)
shuffle_indices = np.random.permutation(np.arange(len(y)))
#pdb.set_trace()
x,y = np.array(x), np.array(y)
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_len = round(len(y)*FLAGS.dev_size_percent)
x_train, x_dev = x_shuffled[:-dev_len], x_shuffled[-dev_len:]
y_train, y_dev = y_shuffled[:-dev_len], y_shuffled[-dev_len:]
print("Dataset Size: {:d}".format(len(y)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement = FLAGS.allow_soft_placement,
      log_device_placement = FLAGS.log_device_placement,
      )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = AudioCNN(
            spect_dim=random.choice(list(spect_dict.values())).shape,
            num_classes=2,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        if FLAGS.l2_constraint: #add l2 constraint as described in (Kim, Y. 2014)
             grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.l2_constraint), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        print(save_flags)
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", ','.join(save_flags)))
        print("Writing to {}\n".format(out_dir))
        
        # Train/Dev Summary Dirs
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch, ):
            '''
            A single training step
            '''
            train_stats = StatisticsCollector()

            feed_dict = {
              cnn.input_song1: tuple(spect_dict[i[0]] for i in x_batch),
              cnn.input_song2: tuple(spect_dict[i[1]] for i in x_batch),
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_factor
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            train_stats.collect(accuracy, loss)

            time_str = datetime.datetime.now().isoformat()
            _, _, summaries = train_stats.report()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_dev, y_dev, writer=None):
            '''
            Evaluates model on full dev set.
            --------------------------------
            Since full dev set likely won't fit into memory, this function
            splits the dev set into minibatches and returns the average 
            of loss and accuracy to cmd line and to summary writer
            '''
            dev_stats = StatisticsCollector()

            dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), 
                                      FLAGS.batch_size, 1)
            for dev_batch in dev_batches:
                if len(dev_batch) > 0:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    feed_dict = {
                      cnn.input_song1: tuple(spect_dict[i[0]] for i in x_dev_batch),
                      cnn.input_song2: tuple(spect_dict[i[1]] for i in x_dev_batch),
                      cnn.input_y: y_dev_batch,
                      cnn.dropout_keep_prob: 1.0
                    }

                    step, loss, accuracy = sess.run(
                        [global_step, cnn.loss, cnn.accuracy],
                        feed_dict)

                    dev_stats.collect(accuracy, loss)

            time_str = datetime.datetime.now().isoformat()
            batch_accuracy, batch_loss, summaries = dev_stats.report()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, batch_loss, batch_accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate training batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                # send entire dev set to dev_step each eval and split into minibatches there
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}".format(path))

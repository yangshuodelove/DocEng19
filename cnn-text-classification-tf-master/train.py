#! /usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)") # 原来 = 128|暂不改为256
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 10, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)") # 原来64
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")# 原来=200；考虑改为1/5/10
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# 返回x_train, y_train, vocab_processor, x_dev, y_dev
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    # x_text是所有的正负文本；y是标签
    x_text, y = data_helpers.load_data_and_labels_2(FLAGS.positive_data_file, FLAGS.negative_data_file)  # 可以使用load_data_and_labels_2
    print("x_text's length: {}, y's length: {}".format(len(x_text), len(y)))  # x_text's length: 10662, y's length: 10662。因为正负文本各有5331个。
    print("x's type: {}, y's type: {}".format(type(x_text), type(y)))  #  x's type: <class 'list'>, y's type: <class 'numpy.ndarray'>

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])  # max_document_length中存放最长句子的长度
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length) # VocabularyProcessor作用是Maps documents to sequences of word ids.
    x = np.array(list(vocab_processor.fit_transform(x_text))) # fit_transform作用是 Learn the vocabulary dictionary and return indexies of words.
    print("x's shape: {}, x[0]: {}".format(x.shape, x[0]))  # x就是文本中单词变成序号的新文本。x's shape:  (10662, 56)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y))) # np.arange返回的是一个min=0/max=len(y)的list，permutation是将这个list中的数字顺序打乱，返回的仍然是这个list.
    x_shuffled = x[shuffle_indices]  # 打乱x的顺序后，得到x_shuffled
    y_shuffled = y[shuffle_indices]  # 打乱y的顺序后，得到y_shuffled

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("x_train's shape:{}".format(x_train.shape)) # x_train's shape:(9596, 56)
    print("y_train's shape:{}".format(y_train.shape)) # y_train's shape:(9596, 2)
    print("x_dev's shape:{}".format(x_dev.shape)) # x_dev's shape:(1066, 56)
    print("y_dev's shape:{}".format(y_dev.shape)) # y_dev's shape:(1066, 2)

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))  # Vocabulary Size: 18758
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))  # Train/Dev split: 9596/1066
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)

        # 考虑如何在session_conf中加入：gpu_options.allow_growth = True;
        # 试试我们之前运行gpu的那个程序是怎么做到的
        # session_conf = tf.ConfigProto()
        # session_conf.allow_soft_placement = True
        # session_conf.log_device_placement = True
        # session_conf.gpu_options.allow_growth = True
        """
        参数说明：
        sequence_length : 句子长度
        num_classes： 分类任务类型个数
        vocab_size：字典中单词个数
        embedding_size：单词向量维度
        filter_sizes： filter尺寸，列表类型
        num_filters： filter个数
        l2_reg_lambda：正则化weight
        """
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                参数说明：
                x_batch: 1个句子
                y_batch: 1个标签
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches：batch_size = 64; num_epochs = 1(原来是200)
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)),
                FLAGS.batch_size,
                FLAGS.num_epochs)
            # Training loop. For each batch...（注意：这里的batches是一批数据，每1个batch=1个句子+1个标签）
            for batch in batches:  # batches会在需要的时候，（自动）随时提供数据，因为data_helpers.batch_iter中使用了yield
                x_batch, y_batch = zip(*batch) # 这里是解开batch
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step) # global_step：Creates a variable to hold the global_step.
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    # 考虑有无地方需要加上这句话：tf.set_random_seed(10)
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)


if __name__ == '__main__':
    tf.app.run()


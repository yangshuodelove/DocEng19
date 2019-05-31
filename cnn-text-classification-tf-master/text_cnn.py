# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    初始化参数说明：
    sequence_length 句子固定长度（不足补全，超过截断）
    num_classes 多分类, 分为几类.
    vocabulary_size 语料库的词典大小, 记为|D|.
    embedding_size 将词向量的维度, 由原始的 |D| 降维到 embedding_size.
    filter_size 卷积核尺寸
    num_filters 卷积核数量
    l2_reg_lambda 正则化系数
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        # 变量input_x存储句子矩阵，宽为sequence_length，长度自适应（=句子数量）；
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # input_y存储句子对应的分类结果，宽度为num_classes，长度自适应；
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # 变量dropout_keep_prob存储dropout参数，
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        # 常量l2_loss为L2正则超参数。
        l2_loss = tf.constant(0.0)

        """
        通过一个隐藏层, 将 one-hot 编码的词 投影 到一个低维空间中. 
        特征提取器，在指定维度中编码语义特征. 这样, 语义相近的词, 它们的欧氏距离或余弦距离也比较近.
        self.W可以理解为词向量词典，存储vocab_size个 大小为embedding_size的词向量，随机初始化为-1~1之间的值；
        self.embedded_chars是输入input_x对应的词向量表示；size：[句子数量, sequence_length, embedding_size]
        self.embedded_chars_expanded是，将词向量表示扩充一个维度（embedded_chars * 1），维度变为[句子数量, sequence_length, embedding_size, 1]，目的是：
                            方便进行卷积（tf.nn.conv2d的input参数为四维变量，见后文）
        函数tf.expand_dims(input, axis=None, name=None, dim=None)：在input第axis位置增加一个维度（dim用法等同于axis，官方文档已弃用）
        """

        # Embedding layer
        # 词向量层   将词 组装成 低维度的向量
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # tf.nn.embedding_lookup(...)方法执行真正的嵌入操作
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        '''
        conv-maxpool-i/filter_shape：卷积核矩阵的大小，包括num_filters个（输出通道数）大小为
            filter_size*embedding_size的卷积核，输入通道数为1；卷积核尺寸中的embedding_size，
            相当于对输入文字序列从左到右卷，没有上下卷的过程。
        conv-maxpool-i/W：卷积核初始化，shape为filter_shape，元素随机生成，正态分布
        conv-maxpool-i/b：偏移量，num_filters个卷积核，故有这么多个偏移量
        conv-maxpool-i/conv：conv-maxpool-i/W与self.embedded_chars_expanded的卷积
        函数tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
            实现卷积计算
        '''
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # 随机生成正太分布
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # 对卷积操作参数说明
                # input:输入的词向量，[句子数（图片数）batch, 句子定长（对应图高）,词向量维度（对应图宽）, 1（对应图像通道数）]
                # filter:卷积核，[卷积核的高度，词向量维度（卷积核的宽度），1（图像通道数），卷积核个数（输出通道数）]
                # strides:图像各维步长,一维向量，长度为4，图像通常为[1, x, x, 1]
                # padding:卷积方式，'SAME'为等长卷积, 'VALID'为窄卷积
                # 输出feature map：shape是[batch, height, width, channels]这种形式
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                '''
                    value：待池化的四维张量，维度是[batch, height, width, channels]
                    ksize：池化窗口大小，长度（大于）等于4的数组，与value的维度对应，
                        一般为[1,height,width,1]，batch和channels上不池化
                    strides:与卷积步长类似
                    padding：与卷积的padding参数类似
                    返回值shape仍然是[batch, height, width, channels]这种形式
                '''
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # 池化后的结果append到pooled_outputs中。对每个卷积核重复上述操作，
                # 故pooled_outputs的数组长度应该为num_filters。
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        '''
        将pooled_outputs中的值全部取出来然后reshape成
            [len(input_x),num_filters*len(filters_size)]，然后进行了dropout层防止过拟合，  
        最后再添加了一层全连接层与softmax层将特征映射成不同类别上的概率  
        2 3   把池化层输出变成一维向量
        '''
        num_filters_total = num_filters * len(filter_sizes)
        # tf.concat(values, concat_dim)连接values中的矩阵，concat_dim指定在哪一维（从0计数）连接。
        # values[i].shape = [D0, D1, ... Dconcat_dim(i), ...Dn]，连接后就是：
        #   [D0, D1, ... Rconcat_dim, ...Dn]。
        # 回想pool_outputs的shape，是存在pool_outputs中的若干种卷积核的池化后结果，维度为
        #  [len(filter_sizes),batch, height, width, channels=1]，
        # 因此连接的第3维为width，即对句子中的某个词，将不同核产生的计算结果（features）拼接起来。
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

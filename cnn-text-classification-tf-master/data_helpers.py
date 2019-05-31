# -*- coding:utf-8 -*-
import numpy as np
import re

# import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 一开始时，postive-data-file和negative-data-file中只有文本，没有标签，我们需要集成这两个文件并添加标签值，
# 最后返回一个 [所有句子 , 对应标签] 的list
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    参数说明：
    data: x_train's shape=(9596, 56); y_train's shape=(9596, 2)
    batch_size = 64
    num_epochs = 1(原来是200)
    """
    data = np.array(data) # data包含x_train和y_train
    data_size = len(data) # data_size有多少行句子
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 # num_batches_per_epoch每次迭代中，需要有多少个批次的数据传入
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices] # shuffled_data包含x_train和y_train
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size # batch_num是第几个批次
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index] # 生成并返回1个批次的数据


# 一开始时，postive-data-file和negative-data-file中只有文本，没有标签，我们需要集成这两个文件并添加标签值，
# 最后返回一个 [所有句子 , 对应标签] 的list
def load_data_and_labels_2(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    positive_examples = [conceptUpdate(s) for s in positive_examples]

    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    negative_examples = [conceptUpdate(s) for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    # print("x_text's type: {}, x_text[0]: {}".format(type(x_text), x_text[0]))
    # for sent in x_text:
    #     print("sent's type: ", type(sent))
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def conceptUpdate(sentence):
    # print("当前句子：", sentence)
    all_words = []
    sentence_tokenized = word_tokenize(sentence)
    for w in sentence_tokenized:
        all_words.append(add_definition_3(w.lower()))
    all_words_str = ' '.join(str(e) for e in all_words)
    return all_words_str

def add_definition_3(w):
    """
    返回单词的concepts
    :param w: 单词
    :return: 返回形式 str([单词，concepts])
    """
    if wordnet.synsets(w) !=[]:  # and len(wordnet.synsets(w))!=0
        syns = wordnet.synsets(w)
        # print(syns[0].definition())
        word_concept = [w] + syns[0].definition().split(" ")
        words_concept_str = ' '.join(str(e) for e in word_concept)
        # 此处需要加入两个句子的语义相似性判断，用于判断应该返回当前单词的哪个concepts
        return words_concept_str  # [w]+syns[0].definition().split(" ")
    else:
        return str(w)  # [w]



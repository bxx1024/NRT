
import pickle
import os
import csv
import tensorflow as tf
import numpy as np
import sys
import re
sys.path.append('./')
from Constants import TPS_DIR, CATEGORY, TIP_LEN



tf.flags.DEFINE_string("train_data", os.path.join(TPS_DIR, CATEGORY + '_train.csv'), "Data for training")
tf.flags.DEFINE_string("valid_data", os.path.join(TPS_DIR, CATEGORY + '_valid.csv'), " Data for validation")
tf.flags.DEFINE_string("test_data", os.path.join(TPS_DIR, CATEGORY + '_test.csv'), "Data for testing")
tf.flags.DEFINE_string("meta_data", os.path.join(TPS_DIR, CATEGORY + '_infor_meta.csv'), "meta_data")

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
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
    return string.strip() if TREC else string.strip().lower()



def pad_tips(sentences, max_len):
    result = []
    for sentence in sentences:
        if max_len - 3 > len(sentence):
            num_padding = max_len - len(sentence) - 2
            new_sentence = ["<START/>"] + sentence + ["<END/>"] + ["<NULL/>"] * num_padding
            result.append(new_sentence)
        else:
            new_sentence = ["<START/>"] + sentence[:max_len - 3] + ["<END/>"] + ["<NULL/>"]
            result.append(new_sentence)
    return result


def build_vocab_tip(y_tip, is_tip = False):
    voc_num = {}
    for x in y_tip:
        for xx in x:
            if xx not in voc_num: voc_num[xx] = 0
            voc_num[xx] += 1

    if is_tip:
        result = [term for term, num in voc_num.items() if num > 5]
    else:
        result= [term for term, num in voc_num.items() if num > 40]
    vocabulary_tip = dict((term, num) for num, term in enumerate(result))
    vocabulary_tip["<END/>"] = len(vocabulary_tip)
    if is_tip:
        vocabulary_tip["<START/>"] = len(vocabulary_tip)
        vocabulary_tip["<NULL/>"] = len(vocabulary_tip)
    return vocabulary_tip


def review_process(y_review):
    reviews = []
    for idx, review in enumerate(y_review):
        review_dict = {}
        for term in review:
            if term not in review_dict: review_dict[term] = 0
            review_dict[term] += 1
        value_sum = sum(review_dict.values())
        review_dict = dict((key, value/value_sum) for key, value in review_dict.items())
        reviews.append(review_dict)
    return reviews

def review_pad(y_review, review_len, review_vocab_size):
    vocab_list, frequency_list = [], []
    for review_dict in y_review:
        vocab = list(review_dict.keys())
        frequency = list(review_dict.values())
        if len(vocab) < review_len:
            vocab = vocab + [review_vocab_size - 1] * (review_len - len(vocab))
            frequency = frequency + [0.0] * (review_len - len(frequency))
        vocab_list.append(np.array(vocab))
        frequency_list.append(np.array(frequency))
    return np.array(vocab_list), np.array(frequency_list)

def load_data(train_data, valid_data, test_data):
    y_train, y_valid, y_test, y_train_tip, y_valid_tip, y_test_tip, \
    y_train_review, y_valid_review, y_test_review, \
    uid_train, iid_train, uid_valid, iid_valid,uid_test, iid_test = load_data_and_labels(train_data, valid_data, test_data)

    user_num = len(pickle.load(open(os.path.join(TPS_DIR, 'user2id'), 'rb')))
    item_num = len(pickle.load(open(os.path.join(TPS_DIR, 'item2id'), 'rb')))
    print("load data done")

    # tip process
    vocabulary_tip = build_vocab_tip(y_train_tip + y_valid_tip + y_test_tip, True)  # tip词集
    tip_idx_vocab = {}
    for words, idx in vocabulary_tip.items(): tip_idx_vocab[idx] = words
    y_train_tip = [[xx for xx in x if xx in vocabulary_tip] for x in y_train_tip]
    y_valid_tip = [[xx for xx in x if xx in vocabulary_tip] for x in y_valid_tip]
    y_test_tip = [[xx for xx in x if xx in vocabulary_tip] for x in y_test_tip]
    max_tip_len = TIP_LEN + 3  # tip_len
    y_train_tip = pad_tips(y_train_tip, max_tip_len)
    y_valid_tip = pad_tips(y_valid_tip, max_tip_len)
    y_test_tip = pad_tips(y_test_tip, max_tip_len)
    y_train_tip = np.array([np.array([vocabulary_tip[word] for word in words]) for words in y_train_tip])
    y_valid_tip = np.array([np.array([vocabulary_tip[word] for word in words]) for words in y_valid_tip])
    y_test_tip = np.array([np.array([vocabulary_tip[word] for word in words]) for words in y_test_tip])
    print("pad tip done")

    # review process
    vocabulary_review = build_vocab_tip(y_train_review, False)  # tip词集
    review_idx_vocab = {}
    for words, idx in vocabulary_review.items(): review_idx_vocab[idx] = words
    y_train_review = [[xx for xx in x if xx in vocabulary_review] for x in y_train_review]
    y_train_review = [[vocabulary_review[word] for word in words] for words in y_train_review]
    y_train_review = review_process(y_train_review)
    review_len = sorted([len(term.keys()) for term in y_train_review])[-1]
    vocab_list, frequency_list = review_pad(y_train_review, review_len, len(vocabulary_review))
    print('review preprocess done')

    return [y_train, y_valid, y_test, y_train_tip, y_valid_tip, y_test_tip, y_train_review,vocab_list, frequency_list,
            vocabulary_tip, tip_idx_vocab, max_tip_len, vocabulary_review, review_idx_vocab,
            uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num, item_num]


def load_data_and_labels(train_data, valid_data, test_data):

    print("training...")
    uid_train, iid_train = [], []
    y_train, y_train_tip, y_train_review = [], [], []
    f_train = csv.reader(open(train_data, "r", encoding = 'utf-8'))
    for line in f_train:
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        y_train.append(float(line[2]))
        y_train_tip.append(line[3].split(" "))
        y_train_review.append(line[4].split(" "))

    print("validing...")
    uid_valid, iid_valid = [], []
    y_valid, y_valid_tip, y_valid_review = [], [], []
    f_valid = csv.reader(open(valid_data, "r", encoding = 'utf-8'))
    for line in f_valid:
        #if int(line[0]) in uid_train and int(line[1]) in iid_train:
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        y_valid.append(float(line[2]))
        y_valid_tip.append(line[3].split(" "))
        y_valid_review.append(line[4].split(" "))

    print("testing...")
    uid_test, iid_test = [], []
    y_test, y_test_tip, y_test_review = [], [], []
    f_test= csv.reader(open(test_data, "r", encoding = 'utf-8'))
    for line in f_test:
        #if int(line[0]) in uid_train and int(line[1]) in iid_train:
        uid_test.append(int(line[0]))
        iid_test.append(int(line[1]))
        y_test.append(float(line[2]))
        y_test_tip.append(line[3].split(" "))
        y_test_review.append(line[4].split(" "))

    return [np.array(y_train), np.array(y_valid), np.array(y_test),
            y_train_tip, y_valid_tip, y_test_tip,
            y_train_review, y_valid_review, y_test_review,
            np.array(uid_train), np.array(iid_train),
            np.array(uid_valid), np.array(iid_valid),
            np.array(uid_test), np.array(iid_test)]




FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

y_train, y_valid, y_test, y_train_tip, y_valid_tip, y_test_tip, y_train_review,vocab_list, frequency_list, \
vocabulary_tip, tip_idx_vocab, max_tip_len, vocabulary_review, review_idx_vocab, \
uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num, item_num\
    = load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data)

y_train = y_train[:, np.newaxis]
uid_train = uid_train[:, np.newaxis]
iid_train = iid_train[:, np.newaxis]

y_valid = y_valid[:, np.newaxis]
uid_valid = uid_valid[:, np.newaxis]
iid_valid = iid_valid[:, np.newaxis]

y_test = y_test[:, np.newaxis]
uid_test = uid_test[:, np.newaxis]
iid_test = iid_test[:, np.newaxis]

print('ziping...')
batches_train = np.array(list(zip(uid_train, iid_train, y_train, y_train_tip, vocab_list, frequency_list)))
batches_test = np.array(list(zip(uid_test, iid_test, y_test, y_test_tip, np.zeros_like(uid_test), np.zeros_like(uid_test))))

para = {}
para['user_num'] = user_num
para['item_num'] = item_num
para['max_tip_len'] = max_tip_len
para['tip_vocab'] = vocabulary_tip
para['tip_idx_vocab'] = tip_idx_vocab
para['tip_vocab_size'] = len(vocabulary_tip)
para['review_vocab'] = vocabulary_review
para['review_idx_vocab'] = review_idx_vocab
para['review_vocab_size'] = len(vocabulary_review)
para['train_len'] = len(y_train)
para['valid_len'] = len(y_valid)
para['test_len'] = len(y_test)

# pickle.dump(para, open(os.path.join(TPS_DIR, CATEGORY + '.para'), 'wb'))







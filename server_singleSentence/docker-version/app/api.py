#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding: utf-8
from flask import Flask

app = Flask(__name__)

import jieba
import regex
import numpy as np
import os
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def word_segment(sent):
    '''
    Args:
      sent: A string. A sentence.

    Returns:
      A list of words.
    '''
    words = list(jieba.cut(sent, cut_all=False))
    return words


def clean_text(text):
    # Common
    text = regex.sub("(?s)<ref>.+?</ref>", "", text)  # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text)  # remove html tags
    text = regex.sub("&[a-z]+;", "", text)  # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text)  # remove markup tags
    text = regex.sub("(?s){.+?}", "", text)  # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text)  # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text)  # remove media links

    text = regex.sub("[']{5}", "", text)  # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text)  # remove bold symbols
    text = regex.sub("[']{2}", "", text)  # remove italic symbols
    text = text.rstrip()

    # Chinese specific
    text = regex.sub("[^\r\n\p{Han}。！？]", "", text)

    # Common
    text = regex.sub("[ ]{2,}", " ", text)  # Squeeze spaces.
    return text


def get_idx_from_sent(sent, word_idx_map, k=300):
    """
    Transforms sentence into a list of indices. 
    """
    x = []
    words = list(jieba.cut(sent, cut_all=False))

    for word in words:

        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x


def make_idx_data_cv(revs, word_idx_map, cv, k=300):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    train_y, test_y = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map, k)

        if rev["split"] == cv:
            test.append(sent)
            test_y.append(rev["y"])
        else:
            train.append(sent)
            train_y.append(rev["y"])
    # train = np.array(train, dtype='int')
    # test = np.array(test, dtype='int')

    return [train, test, train_y, test_y]


def build_label_list(data_folder):
    label_list = []
    for i, name in enumerate(sorted(os.listdir(data_folder))):
        path = os.path.join(data_folder, name)
        # print(path)
        if name == '.DS_Store':
            continue
        else:
            label_list.append(name)
    return label_list


def build_index_label(pred, label_list):
    a = max([(v, i) for i, v in enumerate(pred[0])])
    idx = a[1]
    if a[0] > 0.7:
        return label_list[idx]
    else:
        return '未知分类'


def build_data_text(text, clean_string=True):
    """
    Loads text and returns revs dictionary
    """
    rev = text
    vocab = defaultdict(float)
    if clean_string:
        orig_rev = clean_text(" ".join(rev))
        # print(orig_rev)
    else:
        orig_rev = " ".join(rev)
    words = set(word_segment(orig_rev))

    for word in words:
        # print(word)
        vocab[word] += 1
    datum = {"y": 0,
             "text": orig_rev,
             "num_words": len(words),
             "split": np.random.randint(0, 10)}

    return datum, vocab


def get_idx_from_text(cl_text, word_idx_map):
    """
    Transforms sentence into a list of indices. 
    """
    x = []
    words = list(jieba.cut(cl_text, cut_all=False))

    for word in words:

        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x

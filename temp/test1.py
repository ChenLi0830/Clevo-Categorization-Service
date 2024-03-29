#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding: utf-8
import codecs
import os
import regex
import jieba
print("jieba succesfuly loaded!")
import numpy as np


from collections import defaultdict
import pandas as pd
import csv
import re
from hanziconv import HanziConv # transform traditional to simplified chinese


#from gensim import corpora, models, similarities
#from gensim.models import Word2Vec

def clean_text(text):
    
    
    # Common
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    text = text.rstrip()
    
    # Chinese specific
    text = regex.sub("[^\r\n\p{Han}\d 。！？]", "", text)
    
    
    # Common
    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
    return text

def sentence_segment(text):
    '''
    Args:
      text: A string. A unsegmented paragraph.
    
    Returns:
      A list of sentences.
    '''
    
    # Chinese specific
    sents = regex.split("([。！？])?[\n]+|[。！？]", text) 
    return sents

def word_segment(sent):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A list of words.
    '''
    words = list(jieba.cut(sent, cut_all=False)) 
    return words

def build_data_cv2(data_folder,cv = 3,clean_string= True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    for i,name in enumerate(sorted(os.listdir(data_folder))):
        path = os.path.join(data_folder, name)
        print(path)
        if name == '.DS_Store':
            continue

        with codecs.open(path, "rb",'utf_8') as f:
            for line in f: 
                rev = []
                rev.append(line)

                if clean_string:
                    orig_rev = clean_text(" ".join(rev))
                    #print(orig_rev)
                else:
                    orig_rev = " ".join(rev)
                words = set(word_segment(orig_rev))

                for word in words:
                    #print(word)
                    vocab[word] += 1
                datum  = {"y":i-1, 
                          "text": orig_rev,                             
                          "num_words": len(words),
                          "split": np.random.randint(0,cv)}
                revs.append(datum)
    return revs, vocab




def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    '''
    fname : zh/zh.tsv (chinese)/ GoogleNews-vectors-negative300.bin (english)
    '''
    word_vecs = {}
    with open(fname,'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        
        current_word = []
        last_word = []
        last_vec = np.array([])
        for row in tsvin:
            
            #print 'The row number is', i
            #print 'if the len is 1, means the current word is in this row :', len(row[0])
            #print 'the len of the word vector of the last word is', len(last_vec)

            if (len(row[0])<=6):
                if not last_word:
                    last_word = HanziConv.toSimplified(row[1])
                else:
                    last_word = current_word


                current_word = HanziConv.toSimplified(row[1])

                #print 'last and new arrived current words are :', last_word, current_word
                if current_word != last_word and len(last_vec) == 300:
                    complete_flag = True
                else:
                    complete_flag = False

                if complete_flag:
                    word_vecs[last_word] = last_vec
                    last_vec = np.array([])
                    #print('Successfuly load 1 vector. Dim of a word vector', len(word_vecs[last_word]))

                if current_word in vocab:
                    #print(current_word)
                    #print('current word is in vocab.')
                    read_flag = True
                else:
                    read_flag = False

                if read_flag:
                    # remove [ from the first vector             
                    a = re.sub('\[','',row[2])
                    last_vec = np.append(last_vec, list(map(np.float32, a.split())))
                else:
                    continue

            else:
                if read_flag:
                    a = re.sub('\]','',row[0])
                    last_vec = np.append(last_vec, list(map(np.float32, a.split())))

    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

if __name__=="__main__":    
    w2v_file =  'zh/zh.tsv'   
    data_folder = 'chen_test'#["textFiles/text.S_INTRO", "textFiles/text.S_END","textFiles/text.S_WAIT_WARNING"]     
    print("loading data...")        
    revs, vocab = build_data_cv2(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print(("number of sentences: " + str(len(revs))))
    print(("vocab size: " + str(len(vocab))))
    print(("max sentence length: " + str(max_l)))
    print("loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, vocab)

    


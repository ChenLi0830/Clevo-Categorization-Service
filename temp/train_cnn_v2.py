#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 15:57:40 2017

@author: admin
"""

'''This example demonstrates the use of Convolution1D for text classification.
'''



#import sys
#sys.path.append('/Users/wangwei/anaconda2/envs/python3_keras/lib/python3.6/site-packages')

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

#os.chdir('/Users/wangwei/cuda_keras_projets/keras/examples/')

import six.moves.cPickle as pickle # for python 3
#import cPickle for python 2.7

import pandas as pd
import numpy as np

import jieba


# set parameters:

maxlen = 64 #11
batch_size = 5
embedding_dims = 300
filters = 50 # 100
kernel_size = 3
hidden_dims = 100
epochs = 100

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
    train_y, test_y = [],[]
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map, k)
        
        if rev["split"]==cv:
            test.append(sent)
            test_y.append(rev["y"])
        else:
            train.append(sent)
            train_y.append(rev["y"])
    #train = np.array(train, dtype='int')
    #test = np.array(test, dtype='int')
    
    return [train, test, train_y, test_y]



if __name__=="__main__":    
    print('The script that is running is :', __file__)
    print('Depending on the training datasets: \n maximum length of a sentence is :', maxlen)

    ######### Main code starts here ###########
    print("loading data...")
    x = pickle.load(open("mr_folder/mr.p","rb"), encoding='latin1')
    revs, W, W2, word_idx_map, word_idx_map2, vocab = x[0], x[1], x[2], x[3], x[4],x[5]
    print("data loaded!")
    print("using: word2vec vectors")

    tmp = pd.DataFrame(revs)

    max_l = np.max(tmp["num_words"])
    print("number of sentences: " , str(len(revs)))
    print("vocab size: " , str(len(vocab)))
    print("max sentence length: " + str(max_l))

    max_features = len(vocab)#50

    #### Make datasets
    datasets = make_idx_data_cv(revs, word_idx_map2, 1, k=300)
    x_train = datasets[0]
    x_test = datasets[1]
    y_train = datasets[2]
    y_test = datasets[3]

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    ############# modelling with CNN
    import keras
    num_classes = 9
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('lengh of y_train is :', y_train.shape[0])
    print('Build model...')
    
	    
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(max_features + 1,
                            embedding_dims,
                            weights = [W],
                            input_length = maxlen,
                            trainable = False)
	
    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape= (maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(32, 7, activation='relu')(embedded_sequences)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(32, 4, activation='relu')(x)
    #x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(9, activation='softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

    model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose = 2,
          validation_data=(x_test, y_test))

    
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # serialize model to JSON
    model_json = model.to_json()
    with open("mr_folder/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("mr_folder/model.h5")
    print("Saved model to disk")
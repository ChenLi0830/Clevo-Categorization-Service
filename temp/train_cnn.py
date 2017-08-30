'''This example demonstrates the use of Convolution1D for text classification.
'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
import sys
sys.path.append('/Users/wangwei/anaconda2/envs/python3_keras/lib/python3.6/site-packages')

import os
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
epochs = 30

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
	revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
	print("data loaded!")
	print("using: word2vec vectors")

	tmp = pd.DataFrame(revs)

	max_l = np.max(tmp["num_words"])
	print("number of sentences: " , str(len(revs)))
	print("vocab size: " , str(len(vocab)))
	print("max sentence length: " + str(max_l))

	max_features = len(vocab)#50

	#### Make datasets
	datasets = make_idx_data_cv(revs, word_idx_map, 1, k=300)
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
	num_classes = 3
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	print('lengh of y_train is :', y_train.shape[0])
	print('Build model...')
	model = Sequential()

	# we start off with an efficient embedding layer which maps
	# our vocab indices into embedding_dims dimensions
	model.add(Embedding(max_features+1,
	                    embedding_dims,
	                    weights=[W],
	                    input_length=maxlen,
	                   trainable=False))
	model.add(Dropout(0.2))

	# we add a Convolution1D, which will learn filters
	# word group filters of size filter_length:
	model.add(Conv1D(filters,
	                 kernel_size,
	                 padding='valid',
	                 activation='relu',
	                 strides=1))
	# we use max pooling:
	model.add(GlobalMaxPooling1D())

	# We add a vanilla hidden layer:
	model.add(Dense(hidden_dims))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))

	# We project onto a single unit output layer, and squash it with a sigmoid:
	model.add(Dense(1))
	model.add(Activation('sigmoid'))


	######################
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	# model.compile(loss=keras.losses.categorical_crossentropy,
	#               optimizer=keras.optimizers.Adadelta(),
	#               metrics=['accuracy'])
	model.compile(optimizer='rmsprop', 
	              loss='categorical_crossentropy', 
	              metrics=['accuracy'])
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
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
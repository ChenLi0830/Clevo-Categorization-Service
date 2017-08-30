# encoding: utf-8
import codecs
import os
import regex
import jieba
import numpy as np
import pandas as pd

import six.moves.cPickle as pickle # for python 3
#import cPickle for python 2.7
from process_data import clean_text, sentence_segment, word_segment
from process_data import load_bin_vec, get_W, build_data_cv2, add_unknown_words
from train_cnn import make_idx_data_cv, get_idx_from_sent
from keras.preprocessing import sequence
from keras.models import model_from_json


# ARR = ["您好960号为您服务，请问有什么可以帮您的",
# "您好，642号为您服务",
# "您好956为您服务，请问有什么可以帮您",
# "您好959号为您服务",
# "您好946为您服务，请问有什么可以帮您",
# "您好936为您服务",
# "您好925号为您服务",
# "您好804为您服务，有什么可以帮您",
# "您好671为您服务",
# "您好642号为您服务",
# "您好，966号为您服务，请问有什么可以帮您",
# "您好960号为您服务，请问有什么可以帮您的"]

if __name__=="__main__":    
    # new data path 
    data_folder = '/Users/wangwei/cuda_keras_projets/clevo/CNN_clevo/chen_test'
    
    # Storing them in a dictionary 
    revs, vocab = build_data_cv2(data_folder, cv=10, clean_string=True)
    
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    
   

    # loading word_idx_map for training set
    print("loading word_idx_map data...")
    x = pickle.load(open("mr_folder/mr.p","rb"), encoding='latin1')
    revs, W, W2, word_idx_map, vocab, w2v = x[0], x[1], x[2], x[3], x[4], x[5]

    # Creating datasets
    datasets = make_idx_data_cv(revs, word_idx_map, 1, k=300)
    x_train = datasets[0]
    x_test = datasets[1]
    test_all = x_train + x_test
    print('test_all shape', len(test_all))

    # Padding sequences
    maxlen = 64
    print('Pad sequences (samples x time)')
    x_test = sequence.pad_sequences(test_all, maxlen=maxlen)
    print('x_test shape:', x_test.shape)

    # Loading model
    # load json and create model
    json_file = open('mr_folder/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("mr_folder/model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #score = loaded_model.evaluate(X, Y, verbose=0)
    y_test = loaded_model.predict(x_test)
    print(y_test[:5])
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    
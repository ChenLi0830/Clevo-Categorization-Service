# encoding: utf-8
import regex
import jieba
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import six.moves.cPickle as pickle # for python 3
#import cPickle for python 2.7
from process_data_data1 import clean_text, sentence_segment, word_segment
from process_data_data1 import load_bin_vec, get_W, build_data_cv2, add_unknown_words
from train_cnn_demo import make_idx_data_cv, get_idx_from_sent
from keras.preprocessing import sequence
from keras.models import model_from_json
import keras



if __name__=="__main__":    
    # new data path 
    #data_folder = '/Users/wangwei/cuda_keras_projets/clevo/CNN_clevo/chen_test'
    data_folder = 'data' 
    # Storing them in a dictionary 
    revs, vocab, tags = build_data_cv2(data_folder, cv=10, clean_string=True)
    
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    
   

    # loading word_idx_map for training set
    print("loading word_idx_map data...")
    x = pickle.load(open("mr_folder/mr_demo.p","rb"), encoding='latin1')
    revs, W, W2, word_idx_map, word_idx_map2, vocab = x[0], x[1], x[2], x[3], x[4], x[5]

    # Creating datasets
    datasets = make_idx_data_cv(revs, word_idx_map2, 9, k=300)
    x_train = datasets[0]
    x_test = datasets[1]
    y_train = datasets[2]
    y_test = datasets[3]
    test_all = x_train + x_test
    test_yall = y_train + y_test
    print('test_all shape', len(test_all))
    print('test_yall shape', len(test_yall))

    # Padding sequences
    maxlen = 580
    print('Pad sequences (samples x time)')
    x_test = sequence.pad_sequences(test_all, maxlen=maxlen)
    print('x_test shape:', x_test.shape)
    num_classes = 9
    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(test_yall, num_classes)
    print('y_test shape:', y_test.shape)
    # Loading model
    # load json and create model
    json_file = open('mr_folder/model_demo.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("mr_folder/model_demo.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(x_test, y_test, verbose=1)
    y_test_hat = loaded_model.predict(x_test)
    #print(y_test[:5])
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    a = []
    for i in range(719):
        a.append( max(y_test_hat[i,]))
    
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    a.sort()
    hmean = np.mean(a)
    hstd = np.std(a)
    pdf = stats.norm.pdf(a, hmean, hstd)
    plt.plot(a, pdf) # including h here is crucial
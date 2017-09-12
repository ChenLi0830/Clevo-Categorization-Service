#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding: utf-8

from flask import Flask
from flask import request
application = Flask(__name__)

import os
import imp
from keras import backend as K


def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        imp.reload(K)
        assert K.backend() == backend

# set_keras_backend("theano")
import jieba
import numpy as np
import regex
import os
from collections import defaultdict
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import six.moves.cPickle as pickle # for python 3
#import cPickle for python 2.7
#from process_data import clean_text, word_segment
#from process_data import load_bin_vec, get_W, build_data_cv2, add_unknown_words
#from train_cnn import make_idx_data_cv, get_idx_from_sent

from keras.preprocessing import sequence
from keras.models import model_from_json
#import keras


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
    text = regex.sub("[^\r\n\p{Han}。！？]", "", text)


    # Common
    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
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
def build_label_list(data_folder):
    label_list = []
    for i,name in enumerate(sorted(os.listdir(data_folder))):
        path = os.path.join(data_folder, name)
        #print(path)
        if name == '.DS_Store':
            continue
        else:
            label_list.append(name)
    return label_list

def build_index_label(pred, label_list):
    a = max([(v,i) for i,v in enumerate(pred[0])])
    idx = a[1]

    return label_list[idx][:-4]

def build_data_text(text,clean_string= True):
    """
    Loads text and returns revs dictionary
    """
    rev = text
    vocab = defaultdict(float)
    if clean_string:
        orig_rev = clean_text(" ".join(rev))
        #print(orig_rev)
    else:
        orig_rev = " ".join(rev)
    words = set(word_segment(orig_rev))

    for word in words:
        #print(word)
        vocab[word] += 1
    datum  = {"y":0,
              "text": orig_rev,
              "num_words": len(words),
              "split": np.random.randint(0,10)}

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
# @application.route('/post/<int:post_id>')

@application.route("/",methods=['GET','POST'])
def hello():
    # print(request)
    text = request.form['text'];
    print(text)

    # return "Hello World!"
    # new data path
    #data_folder = 'processedData'
    label_list = ['内部电话.txt',
                  '内部电话出险救援.txt',
                     '内部电话投诉讨论.txt',
                     '出险报案.txt',
                     '客户投诉.txt',
                     '客户投诉车辆反复出问题.txt',
                     '客服呼出不活跃客户回访.txt',
                     '客服呼出信息核实.txt',
                     '客服呼出出险报案回访.txt',
                     '客服呼出投诉处理回访.txt',
                     '未知分类.txt',
                     '询问信息.txt',
                     '询问信息业务细节.txt',
                     '询问信息位置信息.txt',
                     '询问信息活动细节.txt',
                     '询问信息询问电话.txt']
    #text = '你好，有什么可以帮您唉你好，那个我想问一下咱那个润滑里边那个大众4S店。那个。说不定患者那是要遗弃的还是上海大众，嗯高尔夫是一汽大众，售后是87983734。8798。3734723公星期四签嗯，嗯。'
    #text = '你好，什么可以帮您您好！我想问一下别克凯越换一个前保险杠和一个小小时大约多少钱，那个现在我给您一个4S店售后维修的电话，您去咨询一下4S店这个价格吧，好勒好勒，唉您打一下87527681您可以把我手机上吧这个短信那我给您发一个试一下，吧您稍后看一下手机，看看有没有收到这个短信，好了好了，唉，行行好，那现在再见。嗯。'
    #text = '嗯果然是那个请问咱那个嗯请问压力在吗？刘雅丽，噢他那个测试了不是吧？啊行没事我问一下你我这边咱那个润滑剂克服的，那个咱那个和鞋垫不是撤了吗？然后是不是那个电话都没变，但是迁到中山公园那边那边去了是吧？唉对的，唉他那边有那个就是地址，啊具体地址还有那个电话吗？包括店长咱这边有信息吗，有有是吧？适合和小姐一模一样的吗？嗯那个然后电话对电话是养生对电话一样，谁知张龙，然后你记下地址吧是金三维五稍等稍，等我吧把那个表打开，唉你这有这个表吗？就是最新的。没有没有，是吧，我加一下，吧再加上，把那就是那个店叫什么中山店是吧，就叫中山店。在中山公园店中山公园店啊嗯位置是在金三维五。他就在中山公园吗？啊对，他就在中山花园北门斜对面填写。哈哈镜三为我哪里江山为国中山花园北门，中山花园，北门那边，在对面斜对面，没省医院挺近是吧。唉对，应该是。行，先北门斜对面，然后那个电话我看一下，噢邪见噢噢还是张龙就是876687066288是吧，啊对之前电话没换，行行好勒，谢谢哈好好再见好好再见。'
    # text = '嗯。你好老师，我们这里是咏华集团的，呃您是有辆丰田车，是把之前的润滑保养啊是这样的，为了更好地提供服务，呢核对一下您的信息。您当时在店里是登记过这个平邑县的地址是吗？嗯然后车主生日这边登记的是67年9月，嗯车辆年审的时间呢是到18年4月份是吧？保险是也是18年4月到期。呃润滑油这个自驾游活动，啊平时4S店有没有提醒过，也没提醒是吧？噢行那我们输入法总部客服中心，今后车辆如需帮助您可以随时联系，谁知，嗯嗯好那祝您用车愉快，再见，说。啊。'
    # Storing them in a dictionary
    revs, vocab = build_data_text(text, clean_string=True)

    print("data loaded!")
    print("vocab size: " + str(len(vocab)))
    print("sentence length: " + str(revs['num_words']))



    # loading word_idx_map for training set
    print("loading word_idx_map data...")
    #x = pickle.load(open("mr_folder/mr.p","rb"), encoding='latin1')
    word_idx_map2 = pickle.load(open("mr_folder/idx.p","rb"), encoding='latin1')
    #_, W, W2, word_idx_map, word_idx_map2, vocab = x[0], x[1], x[2], x[3], x[4], x[5]

    sent = list(get_idx_from_sent(revs['text'], word_idx_map2))
    x_test = []
    x_test.append(sent)
    # Creating datasets


    # Padding sequences
    maxlen = 580
    print('Pad sequences (samples x time)')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
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
    #score = loaded_model.evaluate(x_test, y_test, verbose=0)
    y_test_hat = loaded_model.predict(x_test)
    #print(y_test[:5])
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    #label_list = build_label_list(data_folder)
    response = build_index_label(y_test_hat, label_list)
    print(response)
    return response

    #import six.moves.cPickle as cPickle
    #cPickle.dump(word_idx_map2, open("mr_folder/idx_py2.p", "wb"), protocol=2)

if __name__ == "__main__":
	#decide what port to run the application in
	# port = int(os.environ.get('PORT', 7271))
    # application.run(host='127.0.0.1', port=port)
	#run the application locally on the givn port
    application.run()
    #optional if we want to run in debugging mode
	#application.run(debug=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding: utf-8
from flask import Flask
from flask import request
from api import *
import os
import six.moves.cPickle as pickle  # for python 3
# import cPickle for python 2.7
from keras.preprocessing import sequence
from keras.models import model_from_json

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@app.route("/", methods=['GET', 'POST'])
def categorization():
    # new data path 
    # data_folder = 'processedData'
    label_list = ['结束询问',
                  '开头语',
                  '扣款话术3联系商家',
                  '收集信息',
                  '等待致谢',
                  '扣款话术1亲友儿童',
                  '问题回答',
                  '等待提示',
                  '扣款话术2号码保护']
    # text = '你好，有什么可以帮您唉你好，那个我想问一下咱那个润滑里边那个大众4S店。那个。说不定患者那是要遗弃的还是上海大众，嗯高尔夫是一汽大众，售后是87983734。8798。3734723公星期四签嗯，嗯。'
    # text = '你好，什么可以帮您您好！我想问一下别克凯越换一个前保险杠和一个小小时大约多少钱，那个现在我给您一个4S店售后维修的电话，您去咨询一下4S店这个价格吧，好勒好勒，唉您打一下87527681您可以把我手机上吧这个短信那我给您发一个试一下，吧您稍后看一下手机，看看有没有收到这个短信，好了好了，唉，行行好，那现在再见。嗯。'
    # text = '嗯果然是那个请问咱那个嗯请问压力在吗？刘雅丽，噢他那个测试了不是吧？啊行没事我问一下你我这边咱那个润滑剂克服的，那个咱那个和鞋垫不是撤了吗？然后是不是那个电话都没变，但是迁到中山公园那边那边去了是吧？唉对的，唉他那边有那个就是地址，啊具体地址还有那个电话吗？包括店长咱这边有信息吗，有有是吧？适合和小姐一模一样的吗？嗯那个然后电话对电话是养生对电话一样，谁知张龙，然后你记下地址吧是金三维五稍等稍，等我吧把那个表打开，唉你这有这个表吗？就是最新的。没有没有，是吧，我加一下，吧再加上，把那就是那个店叫什么中山店是吧，就叫中山店。在中山公园店中山公园店啊嗯位置是在金三维五。他就在中山公园吗？啊对，他就在中山花园北门斜对面填写。哈哈镜三为我哪里江山为国中山花园北门，中山花园，北门那边，在对面斜对面，没省医院挺近是吧。唉对，应该是。行，先北门斜对面，然后那个电话我看一下，噢邪见噢噢还是张龙就是876687066288是吧，啊对之前电话没换，行行好勒，谢谢哈好好再见好好再见。'
    # text = '噢后面那一卡通考试前面那俩不是吧？'
    text = request.form['text'];
    print(text)

    # Storing them in a dictionary 
    revs, vocab = build_data_text(text, clean_string=True)

    print("data loaded!")
    print("vocab size: " + str(len(vocab)))
    print("sentence length: " + str(revs['num_words']))

    # loading word_idx_map for training set
    print("loading word_idx_map data...")
    # x = pickle.load(open("mr_folder/mr_demo.p","rb"), encoding='latin1')
    word_idx_map2 = pickle.load(open("mr_folder/idx_demo.p", "rb"), encoding='latin1')
    # _, W, W2, word_idx_map, word_idx_map2, vocab = x[0], x[1], x[2], x[3], x[4], x[5]

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
    json_file = open('mr_folder/model_demo.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("mr_folder/model_demo.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(x_test, y_test, verbose=0)
    y_test_hat = loaded_model.predict(x_test)
    # print(y_test[:5])
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    # label_list = build_label_list(data_folder)

    response = build_index_label(y_test_hat, label_list)
    print(response)
    return response


if __name__ == "__main__":
    # decide what port to run the app in
    # port = int(os.environ.get('PORT', 7271))
    # app.run(host='127.0.0.1', port=port)
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=80)
    # optional if we want to run in debugging mode
    # app.run(debug=True)

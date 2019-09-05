import os
import json
import requests
import gensim

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
# from bs4 import BeautifulSoup

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
# from word_tokenize.egs.vlsp2013_crf.word_tokenize import word_tokenize
# from word_tokenize.egs.vlsp2013_crf.word_tokenize.model import *
from underthesea import word_tokenize

app = Flask(__name__)

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class SentimenModel:
    def __init__(self):
        self.max_phrase_length = 200 # max phrase length
        self.model = None
        self.word2index = None
    def init_model_architect(self):        
        self.embedding_matrix = np.zeros((439056, 400))
        model = Sequential()
        model.add(Embedding(439056, 400, weights=[self.embedding_matrix], input_length=200, trainable=False))

        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(1, activation='sigmoid'))
        # Adam Optimiser
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def load_model_from_path(self, model_dir_path):
        model_path = os.path.join(model_dir_path, 'lstm_best_weight_70.hdf5')
        word2index_path = os.path.join(model_dir_path, 'word2index.json')

        self.init_model_architect()
        self.model.load_weights(model_path)
        self.graph = tf.get_default_graph()
        with open(word2index_path, 'r') as handle:
            self.word2index = json.load(handle)

    def word_tokenize(self, sentence):
        return word_tokenize(sentence, format='text')

    def _preprocess_inputs(self, comments):
        # one-hot encoding + padding
        encoded_phrases = [[self.word2index.get(word.lower(), 0) for word in self.word_tokenize(comment).split()] for comment in comments]
        padded_phrases = pad_sequences(encoded_phrases, maxlen=self.max_phrase_length, padding='post')
        return padded_phrases

    def predict(self, comments):
        if self.model != None:
            padded_phrases = self._preprocess_inputs(comments)
            with self.graph.as_default():
                probs = self.model.predict(padded_phrases).reshape(1, len(padded_phrases))[0]
            
            comments = [(cm, probs[i]) for i, cm in enumerate(comments)]

        return comments


# def comment_from_page(page_url, max=None):
#     # request page
#     r = requests.get(page_url)
#     page_html = BeautifulSoup(r.text, 'html.parser')

#     # get comment params
#     cmt_input = json.loads(page_html.find('div', {'id':'box_comment_vne'})['data-component-input'])
    
#     # request comment
#     cmt_url = 'https://usi-saas.vnexpress.net/index/get?offset=0&limit=24'
#     cmt_url += '&objectid=' + str(cmt_input['article_id'])
#     cmt_url += '&objecttype=' + str(cmt_input['article_type'])
#     cmt_url += '&siteid=' + str(cmt_input['site_id'])
#     cmt_url += '&categoryid=' + str(cmt_input['category_id'])
#     cmt_url += '&sign=' + str(cmt_input['sign'])

#     cmt_request = requests.get(cmt_url)
#     cmt_response = BeautifulSoup(cmt_request.text, 'html.parser')
    
#     items = json.loads(cmt_response.text)['data']['items']
#     return [[item['comment_id'],
#              item['content'],
#              item['creation_time']]
#              for item in items]


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        input_text = request.form.get('text') 
        print(input_text)
        text, prob = model.predict([input_text])[0]
        print(text, prob)

        result = 'pos' if prob>=0.5 else 'neg'
        return jsonify(result=result)
    except:
        return None

# @app.route('/classify_article', methods=['POST'])
# def classify_article():
#     page_url = request.form.get('page_url')
#     cmt = comment_from_page(page_url, 5)
#     results = []
#     print(cmt)

#     content = [content for _, content, _ in cmt]
#     content_merge = SPECIAL_DELIMETER.join(content)
#     content_merge_tokened = token.tokenizeOneLine(content_merge)

#     for content_tokened in content_merge_tokened.split(SPECIAL_DELIMETER):
#         print(content_tokened)
#         input_vec = vectorize(content_tokened)

#         if all(input_vec):
#             result = SVM_MODEL.predict([input_vec])
#             print(result)
#             result = 'pos' if result[0]==1 else 'neg'

#             results.append(result)
#     print(result)
#     response = {
#         'url': page_url,
#         'cmt': content,
#         'sent': results
#     }
    
#     return jsonify(response)


model = SentimenModel()
model.load_model_from_path('../data/')


# if __name__ == '__main_':
try:
    print('run server')
    app.run(debug=True, port=8081, host='0.0.0.0')
except:
    print('sc stop')
    sc.stop()
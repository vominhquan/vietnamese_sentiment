import os
import json
import pickle
import requests

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn import svm
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from py4j.java_gateway import java_import
from pyspark.mllib.common import _to_java_object_rdd

global W2V_MODEL
global SVM_MODEL
SPECIAL_DELIMETER = ' ------------------------ '

from pyspark import SparkContext
sc = SparkContext.getOrCreate()


app = Flask(__name__)


def vectorize(text):
    sum_vec = np.zeros(100)
    for token in text.split():
        try:
            vec = W2V_MODEL.wv.get_vector(token.lower())
            # print(vec)
            sum_vec += vec
        except:
            pass
    if np.count_nonzero(sum_vec)==0:
        return None
    else:
        return sum_vec

def comment_from_page(page_url, max=None):
    # request page
    r = requests.get(page_url)
    page_html = BeautifulSoup(r.text, 'html.parser')

    # get comment params
    cmt_input = json.loads(page_html.find('div', {'id':'box_comment_vne'})['data-component-input'])
    
    # request comment
    cmt_url = 'https://usi-saas.vnexpress.net/index/get?offset=0&limit=24'
    cmt_url += '&objectid=' + str(cmt_input['article_id'])
    cmt_url += '&objecttype=' + str(cmt_input['article_type'])
    cmt_url += '&siteid=' + str(cmt_input['site_id'])
    cmt_url += '&categoryid=' + str(cmt_input['category_id'])
    cmt_url += '&sign=' + str(cmt_input['sign'])

    cmt_request = requests.get(cmt_url)
    cmt_response = BeautifulSoup(cmt_request.text, 'html.parser')
    
    items = json.loads(cmt_response.text)['data']['items']
    return [[item['comment_id'],
             item['content'],
             item['creation_time']]
             for item in items]


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    input_text = request.form.get('text')
    input_text_tokened = token.tokenizeOneLine(input_text)
    print(input_text_tokened)

    input_vec = vectorize(input_text_tokened)
    print(input_vec)

    if all(input_vec):
        result = SVM_MODEL.predict([input_vec])
        print(result)
        result = 'pos' if result[0]==1 else 'neg'
        return jsonify(result=result)
    else:
        return None

@app.route('/classify_article', methods=['POST'])
def classify_article():
    page_url = request.form.get('page_url')
    cmt = comment_from_page(page_url, 5)
    results = []
    print(cmt)

    content = [content for _, content, _ in cmt]
    content_merge = SPECIAL_DELIMETER.join(content)
    content_merge_tokened = token.tokenizeOneLine(content_merge)

    for content_tokened in content_merge_tokened.split(SPECIAL_DELIMETER):
        print(content_tokened)
        input_vec = vectorize(content_tokened)

        if all(input_vec):
            result = SVM_MODEL.predict([input_vec])
            print(result)
            result = 'pos' if result[0]==1 else 'neg'

            results.append(result)
    print(result)
    response = {
        'url': page_url,
        'cmt': content,
        'sent': results
    }
    
    return jsonify(response)



# Import vnTokenizer from Java
java_import(sc._gateway.jvm, "vn.vitk.tok.Tokenizer")
Tokenizer = sc._jvm.vn.vitk.tok.Tokenizer
dataFolder = os.getcwd() + '/dat/tok'
token = Tokenizer(sc._jsc, dataFolder + "/lexicon.xml", dataFolder + "/regexp.txt")

# Get word2vec model
with open('W2V_model.pickle', 'rb') as handle:
    W2V_MODEL = pickle.load(handle)

# Get SVM model
with open('SVM_model.pickle', 'rb') as handle:
    SVM_MODEL = pickle.load(handle)


# if __name__ == '__main_':
try:
    print('run server')
    app.run(debug=True, port=8081, host='0.0.0.0')
except:
    print('sc stop')
    sc.stop()
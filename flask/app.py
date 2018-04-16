import os
import pickle
import json

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

from pyspark import SparkContext
sc = SparkContext.getOrCreate()


app = Flask(__name__)


def vectorize(text):
    sum_vec = np.zeros(100)
    for token in text.split():
        try:
            vec = W2V_MODEL.wv.get_vector(token.lower())
            print(vec)
            sum_vec += vec
        except:
            pass
    if np.count_nonzero(sum_vec)==0:
        return None
    else:
        return sum_vec


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
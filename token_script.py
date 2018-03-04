import os
import json
import time
from multiprocessing import Pool
from functools import partial
import pandas as pd

import pyspark
from pyspark import SparkContext
from py4j.java_gateway import java_import
from pyspark.mllib.common import _to_java_object_rdd

# Import vnTokenizer from Java
java_import(sc._gateway.jvm, "vn.vitk.tok.Tokenizer")
Tokenizer = sc._jvm.vn.vitk.tok.Tokenizer
dataFolder = os.getcwd() + '/dat/tok'
token = Tokenizer(sc._jsc, dataFolder + "/lexicon.xml", dataFolder + "/regexp.txt")

topics = ['GiaoDuc', 'PhapLuat', 'TheGioi', 'TheThao', 'ThoiSu']

BATCH = 100

# start tokenizing topics one by one
start = time.time()
for topic in topics:
    read_path = os.getcwd() + '/Data/Raw/' + topic
    write_path = os.getcwd() + '/Data/Tokenized/' + topic

    # resume
    list_tokened = os.listdir(write_path)
    list_raw = os.listdir(read_path)

    if len(list_tokened):
        last_file = list_tokened[-1]
        ind = list_tokened.index(last_file)
    else:
        ind = 0

    print(topic + ' Start at ' + str(ind))
    list_raw = list_raw[ind:]
    
    # Batching
    batch_indices = list_raw[0::BATCH]
    print('Number of remaining batches ', len(batch_indices))
    print('Each batch is ', BATCH)
    
    count_null = 0
    count_write = 0
    count = 0
    
    for b in range(len(batch_indices)): # in each batch
        
        # get all filename in its batch
        if b < len(batch_indices) - 1:
            start_index = list_raw.index(batch_indices[b])
            end_index = list_raw.index(batch_indices[b+1])
            list_file_name = list_raw[start_index:end_index]
        else:
            start_index = list_raw.index(batch_indices[b])
            list_file_name = list_raw[start_index:]            
        
        string_batch = ''
        for filename in list_file_name:
            # read data
            with open(read_path+ '/'+ filename,'r') as fd:
                json_data = json.load(fd)

            title = json_data['title']
            title = title.replace('\t', ' ')
            title = title.replace('\n', ' ')

            content = json_data['content']
            content = content.replace('\t', ' ')
            content = content.replace('\n', ' ')

            title = ' '.join(title.split())
            content = ' '.join(content.split())

            if content == title == "":
                count_null+=1
                bug_path = os.getcwd() + '/Data/log/null.txt' 
                with open(bug_path, 'a') as fd:
                    fd.write(read_path + '/' + filename)
                    fd.write('\n')
                continue

            # each file concat content together
            string_batch += filename + ' =========-----=========== '
            string_batch += title + ' '
            string_batch += content
            string_batch += ' ------=====-------------'
        
        # token
        string_batch_toked = token.tokenizeOneLine(string_batch)
        # split out and write to tokenized files
        articles = string_batch_toked.split(' ------=====-------------')
        for article in articles:
            if article:
                article_filename = article.split(' =========-----=========== ')[0].strip()
                article_content = article.split(' =========-----=========== ')[1].strip()
                # write data
                count_write += 1
                with open(write_path+ '/' + article_filename, 'w') as fd:
                    fd.write(article_content)

        count += BATCH
        if count%1000==0:
            print('percent of topic', len(os.listdir(write_path))*100/len(os.listdir(read_path)))
            end = time.time()
            print('count null', count_null)
            print('count write', count_write)
            print('time', end - start)
            print('================')
            count_null = 0
            count_write = 0
            count = 0
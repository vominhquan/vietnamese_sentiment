import os
import time
from multiprocessing import Pool
from functools import partial
import pandas as pd
import json
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
# topics = ['GiaoDuc']

start = time.time()
for topic in topics:
    count = 0

    read_path = os.getcwd() + '/Data/Raw/' + topic
    write_path = os.getcwd() + '/Data/Tokenized/' + topic
    
    tmp_arr = []
    
    # resume
    list_tokened = os.listdir(write_path)
    list_raw = os.listdir(read_path)
    
    if(len(list_tokened)):
        last_file = list_tokened[-1]
        ind = list_tokened.index(last_file)
    else:
        ind = 0
        
    print(topic + ' Start at ' + str(ind))
    
    for filename in list_raw[ind:]:
        
#         try:
        
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

        if(content==title==""):
            bug_path = os.getcwd() + '/Data/log/null.txt' 
            with open(bug_path,'a') as fd:
                fd.write(read_path+ '/'+ filename)
                fd.write('\n')
            continue
            
        # tokenize
        title_tokened = token.tokenizeOneLine(title)
        content_tokened = token.tokenizeOneLine(content)
        title_tokened += content_tokened
        
        tmp_arr.append((filename, title_tokened))
        
        count += 1
        
        if(count%100==0):
            # write data
            for fn, data in tmp_arr:
                with open(write_path+ '/' + fn,'w') as fd:
                    fd.write(data)
            tmp_arr = []
                
        if(count%1000==0):
            print(count, count*100/len(list_raw))
            end = time.time()
            print(end - start)
#         except:
#             bug_path = os.getcwd() + '/Data/log/except.txt' 
#             with open(bug_path,'a') as fd:
#                 fd.write(read_path+ '/'+ filename)
#                 fd.write('\n')
        

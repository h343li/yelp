# Replace symbols with space
# Tokenize, split by space
# Remove stopwords and other unwanted characters (Can use countvectorizer)
# Stemming/reduce space
# Extract key feature
# Vectorize
# Assign sentiment score
# Compute average sentiment

import pandas as pd
import re
import nltk.data
import numpy
import math
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def det_lang(astring):
    try:
        return detect(astring)
    except:
        pass

def extract_key_text(comment,long, thres1,thres2):
    if len(comment) > long:
        key_comment = comment[0:math.floor(thres2/2)] + comment[-math.ceil(thres2/2):]
    elif len(comment) > thres1:
        key_comment = comment[0:math.floor(thres1/2)] + comment[-math.ceil(thres1/2):]
    else:
        key_comment = comment
    return key_comment

def get_key_comment(filename, filebus, long, thres1,thres2):
    file = pd.read_csv(filename)
    bus_file = pd.read_csv(filebus)
    data = file.merge(bus_file, left_on = 'business_id', \
        right_on = 'business_id',how='left')
    data['text_sentence'] = data['text'].apply(lambda x: \
        tokenizer.tokenize(x.lower().replace('\n',' ')))
    data = data[data['categories'].apply(lambda x: ('Restaurants' in str(x)) or ('Food' in str(x)) or ('Bars' in str(x)))]

    # Detect languages

    data['language'] = data['text'].apply(lambda x:det_lang(x))

    # Only extract English comments
    data_eng = data[data['language'] == 'en']

    # Check length of comment
    data_eng['text_length'] = data_eng['text_sentence'].apply(len)
    data_eng['text_length'].value_counts()

    check = data_eng[data_eng['text_length'] == 9]

# extract_key_text inputs a dataframe of list of sentences and returns core
# sentences specified by thres.


    data_eng['key_comment'] = data_eng['text_sentence'].apply(lambda x: \
        extract_key_text(x,long,thres1,thres2))
    columns_to_keep = ['name', 'text', 'key_comment', 'text_length', 'review_count']
    data_final = data_eng[columns_to_keep]
    return data_final

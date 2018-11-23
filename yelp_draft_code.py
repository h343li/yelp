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

data_one = pd.read_csv('current.csv')
business = pd.read_csv('yelp_business.csv')
data_ori = data_one.merge(business, left_on = 'business_id', right_on = 'business_id',how='left')

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
data_ori['text_sentence'] = data_ori['text'].apply(lambda x: \
    tokenizer.tokenize(x.lower().replace('\n',' ')))
# data['text_sentence'] = data['text'].apply(lambda x: re.split(r"\.|\?|\!",x))
# print(data['text_sentence'].head())

# Detect languages
def det_lang(astring):
    try:
        return detect(astring)
    except:
        pass

data_ori['language'] = data_ori['text'].apply(lambda x:det_lang(x))
print(data_ori['language'].value_counts())

# Only extract English comments
data = data_ori[data_ori['language'] == 'en']

# Check length of comment
data['text_length'] = data['text_sentence'].apply(len)
data['text_length'].value_counts()

print(data['text_length'].value_counts())
# print(data['text_length'].head())

check = data[data['text_length'] == 9]
print(check['text_sentence'].iloc[0])

# extract_key_text inputs a dataframe of list of sentences and returns core
# sentences specified by thres.
def extract_key_text(comment,thres):
    if len(comment) > thres:
        key_comment = comment[0:math.floor(thres/2)] + comment[-math.ceil(thres/2):]
    else:
        key_comment = comment
    return key_comment

data['key_comment'] = data['text_sentence'].apply(lambda x: extract_key_text(x,6))

columns_to_keep = ['name', 'text', 'key_comment', 'text_length', 'review_count']

data_drop = data[columns_to_keep]


print(data_drop.head())

# coding: utf-8

# aspect based opinion-mining using LDA-based models
# - models the targetopinion interaction directly
# - e.g. ‘grilled’ is positive for sausage'
# - automatic learning of the lexicon strength based on the data
# 1. determine aspects from soft clustering
# 

# In[34]:


import pandas as pd
data = pd.read_csv('/Users/thecoolestman/yelp/current.csv', error_bad_lines=False);
data_text = data[['text']]
data_text['index'] = data_text.index
documents = data_text


# In[39]:


print(len(documents))
print(documents[:5])


# In[71]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


# In[68]:


stemmer = SnowballStemmer("english")
stop_no_apos = ['dont', 'youre', 'were', 'hes', 'shes', 'your']
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_no_apos            and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[69]:


processed_docs = documents['text'].map(preprocess)
processed_docs[:10]


# In[128]:


# food dict contains specific food names
# we also need a more generalized dictionary 
count = 0
categories = ['meat.n.01', 'food.n.01']
dictionary = []
for category in categories:
    subcategory = wn.synset(category)
    hypo = lambda s: s.hyponyms()
    subnet = list(subcategory.closure(hypo))
    for words in subnet:
        dictionary.append(words.name().partition('.')[0])

#dictionary = set([w for s in food.closure(lambda s:s.hyponyms()) for w in s.lemma_names()])
#for k, v in dictionary.iteritems():
#    print(k, v)
#    count += 1
#    if count > 10:
#        break


# In[129]:


print(dictionary)


# In[130]:


'rib' in food_dict



# coding: utf-8

# aspect based opinion-mining using LDA-based models
# - models the targetopinion interaction directly
# - e.g. ‘grilled’ is positive for sausage'
# - automatic learning of the lexicon strength based on the data
# 1. determine aspects from soft clustering
# 

# In[1]:


import pandas as pd
data = pd.read_csv('~/current.csv', error_bad_lines=False);
data_text = data[['text']]
data_text['index'] = data_text.index
documents = data_text


# In[2]:


print(len(documents))
print(documents[:5])


# In[3]:


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


# In[119]:


stemmer = SnowballStemmer("english")
stop_no_apos = ['dont', 'youre', 'were', 'hes', 'shes', 'your']
stop_descriptive = ['okay', 'best', 'better', 'good', 'great', 'like', 'go', 'love', 'time', 'nice', 'come',                     'disappoint', 'amaz', 'sunday', 'saturday', 'call', 'cool', 'tell', 'rude', 'buy', 'leav',                    'free', 'hour', 'awesom', 'work', 'littl', 'includ', 'cute', 'friend', 'family', 'feel',                    'guy', 'help', 'want', 'peopl', 'staff', 'give', 'get', 'right', 'happi', 'pretti', 'year',                    'order', 'busi']
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_no_apos            and len(token) > 3:
            post_stem = lemmatize_stemming(token)
            if post_stem not in stop_descriptive:
                result.append(post_stem)
    return result


# In[232]:


processed_docs = documents['text'].map(preprocess)
dictionary = gensim.corpora.Dictionary(processed_docs)
# maybe consider appending this dictonary to wordnet
# Now we filter out tokens that appear in
# - less than 15 documents (absolute number) or
# - more than 0.5 documents (fraction of total corpus size, not absolute number).
dictionary.filter_extremes(no_below=5, no_above=0.7)


# In[231]:


docs = ["The pizza was okay." "Not the best I've had." "I prefer Biaggio's on Flamingo / Fort Apache." "The chef there can make a MUCH better NY style pizza." "The pizzeria @ Cosmo was over priced for the quality and lack of personality in the food." "Biaggio's is a much better pick if youre going for italian - family owned, home made recipes, people that actually CARE if you like their food." "You dont get that at a pizzeria in a casino." "I dont care what you say..."]
texts = [[i for i in doc.lower().split()] for doc in docs]
dictionary = corpora.Dictionary(texts)
dictionary[46]


# In[164]:


# food dict contains specific food names
# we also need a more generalized dictionary 

categories = ['meat.n.01', 'food.n.01']
food_dict = []
for category in categories:
    subcategory = wn.synset(category)
    hypo = lambda s: s.hyponyms()
    subnet = list(subcategory.closure(hypo))
    for words in subnet:
        food_dict.append(words.name().partition('.')[0])
#dictionary.add_documents(food_dict)

#dictionary = set([w for s in food.closure(lambda s:s.hyponyms()) for w in s.lemma_names()])
#for k, v in dictionary.iteritems():
#    print(k, v)
#    count += 1
#    if count > 10:
#        break


# In[233]:


#create a dictionary reporting how many words and how many times those words appear.
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
#create tf-idf model object
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


# In[265]:


# Running LDA using TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=18,                                              id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# In[266]:


for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
import numpy as np
from gensim import corpora, models
np.random.seed(2018)
import nltk
nltk.download('wordnet')

stemmer = SnowballStemmer("english")
stop_no_apos = ['dont', 'youre', 'were', 'hes', 'shes', 'your', 'good', 'great', 'like', 'worth', 'come',\
                'wicked', 'spoon', 'wasn', 'recommend', 'went', 'times', 'selection', 'place', 'coming',\
               'better', 'definitely', 'best', 'time', 'wait']
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_word \
            and len(token) > 3:
            #post_stem = lemmatize_stemming(token)
            post_stem = token
            result.append(post_stem)
    return result

# rev_list: a list of file names containing sentiment-scored reviews
# senti_range_low: lower bound of sentiment level
# senti_range_high: upper bound of sentiment level
# name: string of restaurant of which we perform LDA on
# example:
# res = lda_run(['~/yelp/yelpFinishedchunk2.csv','~/yelp/yelpFinishedchunk4.csv','~/yelp/yelpFinishedchunk13.csv', \
#        '~/yelp/yelpFinishedchunk21.csv','~/Downloads/data_nostar.csv','~/Downloads/data_sofar_Alicia.csv', \
#        '~/Downloads/data_star.csv'], 0.5, 100, 'SkinnyFATS')
def lda_run(rev_list, senti_range_low, senti_range_high, name):
    data = pd.read_csv(rev_list[0], error_bad_lines=False)
    filtered = data[data['sentiment'] > senti_range_low]
    filtered = data[data['sentiment'] < senti_range_high]
    filtered = filtered[filtered['name'] == name]
    data_text = filtered[['text']]
    for file in range(1, len(rev_list)):
        data = pd.read_csv(rev_list[file], error_bad_lines=False)
        filtered = data[data['sentiment'] > senti_range_low]
        filtered = data[data['sentiment'] < senti_range_high]
        filtered = filtered[filtered['name'] == name]
        tmp = filtered[['text']]
        data_text = pd.concat([data, tmp])
    data_text['index'] = data_text.index
    documents = data_text
    processed_docs = documents['text'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    # Now we filter out tokens that appear in
    # - less than 15 documents (absolute number) or
    # - more than 0.5 documents (fraction of total corpus size, not absolute number).
    dictionary.filter_extremes(no_below=10, no_above=0.7)
    #create a dictionary reporting how many words and how many times those words appear.
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    # Running LDA using TF-IDF
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=50,  \
                      id2word=dictionary, passes=2, workers=4)
    all_topics = {}
    for corpus in range(len(bow_corpus)):
        for index, score in sorted(lda_model_tfidf[bow_corpus[corpus]], key=lambda tup: -1*tup[1]):
            list_of_topics = lda_model_tfidf.print_topic(index, 10).split('+')
            for topic in list_of_topics:
                clean = topic.strip()
                i = clean.split('*')
                if i[1] not in list(all_topics.keys()):
                    all_topics[i[1]] = round(float(i[0]), 4)
                else:
                    all_topics[i[1]] = round(float(all_topics[i[1]]) + float(i[0]), 4)
    # normalize
    total = sum(all_topics.values(), 0.0)
    all_topics = {k.replace('"', ''): v / total for k, v in all_topics.items()}
    # sort
    sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse = True)
    print(sorted_topics[0:10])
    return sorted_topics

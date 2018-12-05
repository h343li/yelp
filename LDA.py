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
stop_word = ['dont', 'youre', 'were', 'hes', 'shes', 'your', 'wasn', 'want']
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_word \
            and len(token) > 3:
            post_stem = lemmatize_stemming(token)
            result.append(post_stem)
    return result

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

# rev_list: a list of reviews with the same sentiment score
def lda_run(rev_list):
    data_text = pd.DataFrame(rev_list, columns = ['text'])
    data_text['index'] = data_text.index
    documents = data_text
    processed_docs = documents['text'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    # Now we filter out tokens that appear in
    # - less than 15 documents (absolute number) or
    # - more than 0.5 documents (fraction of total corpus size, not absolute number).
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    #create a dictionary reporting how many words and how many times those words appear.
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    # Running LDA using TF-IDF
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=18,  \
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

    sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse = True)
    print(sorted_topics)
    return sorted_topics

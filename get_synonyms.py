from nltk.corpus import wordnet as wn
import pandas as pd
from itertools import zip_longest
from anytree import Node, RenderTree
import pickle

class senti_word:
    def __init__(self, word, sentiment, parent = None):
        if parent == None:
            self.word_tree = Node([word,sentiment])
        self.sentiment = sentiment
        self.child = self.get_syn(word)
        for child in self.child:
            if parent == None:
                Node(child, parent = self.word_tree)
            else:
                Node(child, parent = parent)

    @staticmethod
    def get_syn(word):
        synonyms = []
        antonyms = []
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
            for s in syn.similar_tos():
                synonyms.append(s.name().split('.')[0])
        return list(set(list(map(lambda x: x.lower().replace("_"," "), synonyms))))

class senti_dictionary:

    def __init__(self, in_file):
        with open(in_file, 'rb') as input:
            self.cur_dict = pickle.load(input)

    def __exists__(self, word):
        if sentiment == 'Positive':
            return (word in self.pos)
        elif sentiment == 'Negative':
            return (word in self.neg)
        elif sentiment == 'Neutral':
            return (word in self.ntl)
        else:
            exit('Wrong sentiment')

    def export(self, pkl_file):
        with open(pkl_file,'wb') as output:
            pickle.dump(self.cur_dict,output)


    def add_to_dict(self, new_word, sentiment, get_tree = True, num_layer = 1):
        tmp = self.cur_dict.keys()
        current_roots = list(map(lambda x: x[0],tmp))
        if new_word in current_roots:
            exit('Already existed!')
        if get_tree:
            newWord = senti_word(new_word, sentiment)
            word_tree = newWord.word_tree
            while num_layer > 1:
                for child in word_tree.children:
                    senti_word(child.name,sentiment,child)
                num_layer = num_layer - 1

            self.cur_dict[new_word] = word_tree
        else:
            self.cur_dict[new_word] = Node([new_word,sentiment])

##############################################################################
# Dictionary Construction ####################################################
##############################################################################
word = 'ok'

yelp_senti = senti_dictionary(in_file = "SentiForest.pkl")
yelp_senti.add_to_dict('one-star','Negative',False,2)
yelp_senti.add_to_dict('under-trained','Negative',False,2)
yelp_senti.add_to_dict('stay-away','Positive',False,2)
yelp_senti.add_to_dict('above-average','Positive',False,2)
yelp_senti.add_to_dict('die-for','Positive',False,2)
yelp_senti.add_to_dict('below-average','Positive',False,2)
yelp_senti.export('SentiForest.pkl')

neutral = ['ok','okay','alright','expected','common','acceptable','satisfactory',\
'lean','fine','reasonable','average','soft','crunchy','smoky','sweet']
positive = ['good','great','nice','friendly','recommend','amazing','better','new','delicious','clean',\
'happy','fresh','different','fantastic','wonderful','fast','professional','cheap','comfortable',\
'authentic','pleasant','helpful','beautiful', 'warm', 'impressive','impressed','fabulous', \
'yummy', 'excellent', 'extensive','tender', 'favourite', 'favorite', 'like','flavorful','thanks', \
'convenient','memorable','indulgent','attentive','spacious','paramount','worth']
negative = ['bad','nasty','rude','mean','worse','old','unhappy','mad','dirty','slow','mushy','unprofessional',\
'expensive','uncomfortable','ugly','unhelpful','unless','awful','poor','disappoint','disappointment','mundane', 'fat', \
'terrible', 'ignore','hate','lack', 'poison', 'inferior', 'late', 'disrespectful','misleading',\
'lacking','horrible','miserable','sketchy','stale','rancid','sour','inedible','dry','unplatable', \
'dull','insipid','pedestrian','mediocre','annoying','underwhelming','unattentive', \
'ridiculous','garbage','overcooked','embarrass','waste','lack','mess']

for word in neutral:
    yelp_senti.add_to_dict(word, 'Neutral', True, 2)
    yelp_senti.export('SentiForest.pkl')

for word in positive:
    yelp_senti.add_to_dict(word, 'Positive', True, 2)
    yelp_senti.export('SentiForest.pkl')

for word in negative:
    yelp_senti.add_to_dict(word, 'Negative', True, 2)
    yelp_senti.export('SentiForest.pkl')

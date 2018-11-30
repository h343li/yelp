from nltk.corpus import wordnet as wn
import pandas as pd
from itertools import zip_longest


class senti_dictionary:

    def __init__(self, in_file):
        current_dict = pd.read_csv(in_file)
        self.pos = current_dict['Positive'][~pd.isna(current_dict['Positive'])].tolist()
        self.neg = current_dict['Negative'][~pd.isna(current_dict['Negative'])].tolist()
        self.ntl = current_dict['Neutral'][~pd.isna(current_dict['Neutral'])].tolist()

    def __exists__(self, word, sentiment):
        if sentiment == 'Positive':
            return (word in self.pos)
        elif sentiment == 'Negative':
            return (word in self.neg)
        elif sentiment == 'Neutral':
            return (word in self.ntl)
        else:
            exit('Wrong sentiment')

    def export(self, out_file):
        new_dict = pd.concat([pd.Series(self.pos, name = 'Positive'), \
        pd.Series(self.neg, name = 'Negative'), \
        pd.Series(self.ntl, name = 'Neutral')], axis = 1)
        new_dict.to_csv(out_file, index = False)

    def add_to_dict(self, parent_word, sentiment, num_layer = 0):

        parent_syn = self.get_syn(parent_word)['synonyms']
        parent_ant = self.get_syn(parent_word)['antonyms']

        for layer in range(num_layer):
            pos_lst = list(map(lambda x: self.get_syn(x)['synonyms'], parent_syn))
            parent_syn = [item for sublist in pos_lst for item in sublist]
            parent_syn = list(set(parent_syn))
            neg_lst = list(map(lambda x: self.get_syn(x)['synonyms'], parent_ant))
            parent_ant = [item for sublist in neg_lst for item in sublist]
            parent_ant = list(set(parent_ant))

        if sentiment == 'Positive':
            self.pos = list(set(self.pos + parent_syn))
            self.neg = list(set(self.neg + parent_ant))
        elif sentiment == 'Negative':
            self.neg = list(set(self.neg + parent_syn))
            self.pos = list(set(self.pos + parent_ant))
        elif sentiment == 'Neutral':
            self.ntl = list(set(self.ntl + parent_syn))
        else:
            exit('Wrong sentiment')

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
        return {'synonyms':list(set(list(map(lambda x: x.lower().replace("_"," "), synonyms)))),
            'antonyms':list(set(list(map(lambda x: x.lower().replace("_"," "), antonyms))))}

##############################################################################
# Dictionary Construction ####################################################
##############################################################################
yelp_senti = senti_dictionary(in_file = "yelpSentiWordNet.csv")

neutral = ['ok','okay','alright','expected','common','acceptable','satisfactory','fat',\
'dry','lean','fine']
positive = ['good','great','nice','friendly','recommend','amazing','better','new','delicious','clean',\
'happy','fresh','different','fantastic','wonderful','fast','professional','cheap','comfortable',\
'authentic','pleasant','helpful','beautiful']
negative = ['bad','nasty','rude','mean','worse','old','unhappy','mad','dirty','slow','unprofessional'\
'expensive','uncomfortable','ugly','unhelpful','unless']

for word in neutral:
    yelp_senti.add_to_dict(parent_word = word, sentiment = 'Neutral')

for word in positive:
    yelp_senti.add_to_dict(parent_word = word, sentiment = 'Positive',num_layer = 1)

for word in negative:
    yelp_senti.add_to_dict(parent_word = word, sentiment = 'Negative',num_layer = 1)


yelp_senti.export(out_file = "yelpSentiWordNet.csv")

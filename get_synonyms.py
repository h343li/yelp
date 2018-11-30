from nltk.corpus import wordnet as wn
import pandas as pd

class senti_dictionary:

    def __init__(self, in_file):
        current_dict = pd.read_csv(in_file)
        self.pos = current_dict['Positive'].tolist()
        self.neg = current_dict['Negative'].tolist()

    def __exists__(self, word, sentiment):
        if sentiment == 'Positive':
            return (word in self.pos)
        elif sentiment == 'Negative':
            return (word in self.neg)
        else:
            exit('Wrong sentiment')

    def export(self, out_file):
        new_dict = pd.concat([pd.Series(self.pos, name = 'Positive'), pd.Series(self.neg, name = 'Negative')],axis = 1)
        new_dict.to_csv(out_file, index = False)

    def add_to_dict(self, parent_word, sentiment, num_layer = 1):

        parent_syn = self.get_syn(parent_word)['synonyms']
        parent_ant = self.get_syn(parent_word)['antonyms']

        for layer in range(num_layer):
            pos_lst = list(map(lambda x: self.get_syn(x)['synonyms'], parent_syn))
            parent_syn = [item for sublist in pos_lst for item in sublist]
            parent_syn = list(set(list(map(lambda x: x.lower().replace("_"," "), \
                parent_syn))))
            neg_lst = list(map(lambda x: self.get_syn(x)['synonyms'], parent_ant))
            parent_ant = [item for sublist in neg_lst for item in sublist]
            parent_ant = list(set(list(map(lambda x: x.lower().replace("_"," "), \
                parent_ant))))

        if sentiment == 'Positive':
            self.pos = list(set(self.pos + parent_syn))
            self.neg = list(set(self.neg + parent_ant))
        elif sentiment == 'Negative':
            self.neg = list(set(self.neg + parent_syn))
            self.pos = list(set(self.pos + parent_ant))
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
        return {'synonyms':list(set(synonyms)),\
            'antonyms':list(set(antonyms))}

yelp_senti = senti_dictionary(in_file = "yelpSentiWordNet.csv")
yelp_senti.add_to_dict(parent_word = 'different', sentiment = 'Positive')
yelp_senti.export(out_file = "yelpSentiWordNet.csv")

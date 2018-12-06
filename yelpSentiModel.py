# Establish a Senti Class to parse info from SentiWordNet and compute score
# for each given sentences

import nltk
import numpy as np
import re
from stanfordnlp_class import StanfordNLP
import pandas as pd
from get_synonyms import senti_word, senti_dictionary

impt = set(['NNS', 'NN', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
            'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN',
            'VBP', 'VBZ', 'unknown'])
non_base = set(['VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNS', 'NNPS'])
negations = set(['not', 'n\'t', 'less', 'no', 'never',
                'nothing', 'nowhere', 'hardly', 'barely',
                'scarcely', 'nobody', 'none', 'dont',
                'cant', 'couldnt', 'no one',
                'wont','wouldnt', 'doesnt'])
#amplifier = set(['very', 'ever', 'always', 'super', '!', 'fucking',
                #'damn', 'even'])
stopwords_defined = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                        'ourselves', 'you', "you're", "you've", "you'll",
                        "you'd", 'your', 'yours', 'yourself', 'yourselves',
                        'he', 'him', 'his', 'himself', 'she', "she's", 'her',
                        'hers', 'herself', 'it', "it's", 'its', 'itself',
                        'they', 'them', 'their', 'theirs', 'themselves', 'what',
                        'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
                        'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'having', 'do', 'does',
                        'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or',
                        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                        'with', 'about', 'against', 'between', 'into', 'through',
                        'during', 'before', 'after', 'above', 'below', 'to', 'from',
                        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                        'again', 'further', 'then', 'once', 'here', 'there', 'when',
                        'where', 'why', 'how', 'any', 'both', 'each', 'few',
                        'other', 'some', 'such', '.', '!', '?', ';', ':', ','])

def flatten_tree(filename):
    '''inputs filename, a pickle directory of senti_dictionary and returns a data frame of word, sentiment,
    layer, and parent'''
    sentiDict = senti_dictionary(filename)
    col_name = ['word', 'sentiment', 'layer', 'parent']
    senti_df = pd.DataFrame(columns = col_name)
    for key, tree in sentiDict.cur_dict.items():
        root_word = tree.name[0]
        tree_senti = tree.name[1]
        senti_df.loc[senti_df.shape[0]] = [root_word, tree_senti, 0, 'root']
        for layer_one in tree.children:
            senti_df.loc[senti_df.shape[0]] = [layer_one.name, tree_senti, 1, root_word]
            for layer_two in layer_one.children:
                senti_df.loc[senti_df.shape[0]] = [layer_two.name, tree_senti, 2, layer_one.name]
    return senti_df

def pos_short(self,pos):
    # Convert from NLTK POS into SWN POS
    if pos in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
        return 'v'
    elif pos in set(['JJ', 'JJR', 'JJS']):
        return 'a'
    elif pos in set(['RB', 'RBR', 'RBS']):
        return 'r'
    elif pos in set(['NNS', 'NN', 'NNP', 'NNPS']):
        return 'n'
    else:
        return 'a'

class yelpSentiModel(object):
    '''A class defined obtain sentiment score based on analyser'''

    def __init__(self, filename='SentiForest.pkl',inten_file = 'intensifier_dic.txt', \
    emo_file = 'AFINN_emoticon.txt', conjuction_file = 'conjunction_word.txt'):
        flatten_dict = flatten_tree(filename)
        # parse file and build sentiwordnet dicts
        self.intensifier = {}
        self.emoticon = {}
        self.conjunction_word = [line.replace('\n','')  \
            for line in open(conjuction_file)]
        self.pos = flatten_dict[flatten_dict['sentiment'] == 'Positive']
        self.neg = flatten_dict[flatten_dict['sentiment'] == 'Negative']
        self.ntl = flatten_dict[flatten_dict['sentiment'] == 'Neutral']
        self.build_emoticondict(emo_file)
        self.build_intensifier(inten_file)
        print('Initialized!')

    def build_intensifier(self,inten_file):
        '''Extract intensifier factor for sentence scoring'''
        intensify = [line.split() for line in open(inten_file)]
        for pair in intensify:
            word = pair[0]
            multiplier = float(pair[1])
            self.intensifier[word] = multiplier

    def build_emoticondict(self, emo_file):
        '''Extract emoticon factor for sentence scoring'''
        record = [line.split() for line in open(emo_file)]
        for pair in record:
            word = pair[0]
            score = float(pair[1])
            self.emoticon[word] = score

    def score(self, sentence):
        '''Sentiment score of a given sentence, assuming sentence has been tokenzied'''

        wnl = nltk.WordNetLemmatizer()
        sNLP= StanfordNLP()
        sentence_token = nltk.word_tokenize(sentence)
        tagger = sNLP.pos(sNLP.to_truecase(sentence))

        # Assume that sentence is already in tokenized form
        scores = []

        index = 0
        for el in tagger:
            intensifier = 1
            pos = el[1]
            try:
                #word = re.match('(\w+)',el[0]).group(0).lower()
                word = el[0].lower()

                # Check if multi-words

                word_minus_one = sentence_token[max(index-1,0):index+1]
                word_minus_two = sentence_token[max(index-2,0):index+1]
                word_minus_three = sentence_token[max(index-3,0):index+1]

                if (self.is_multiword(word_minus_three)) and len(word_minus_two) == 4:
                    del sentence_token[max(index-3,0):index+1]
                    sentence_token.append('-'.join(word_minus_three))
                    index = index - 3
                    pos = 'unknown'
                    word = '-'.join(word_minus_three)
                elif (self.is_multiword(word_minus_two)) and len(word_minus_two) == 3:
                    del sentence_token[max(index-2,0):index+1]
                    sentence_token.append('-'.join(word_minus_two))
                    index = index - 2
                    pos = 'unknown'
                    word = '-'.join(word_minus_two)
                elif (self.is_multiword(word_minus_one)) and len(word_minus_one) == 2:
                    del sentence_token[max(index-1,0):index+1]
                    sentence_token.append('-'.join(word_minus_one))
                    index = index - 1
                    pos = 'unknown'
                    word = '-'.join(word_minus_one)

                neighborhood = self.find_neighbour(sentence_token,index)

                # look for trailing multiword expressions

                # if multiword expression, fold to one expression
                # Check multi-Senti words

                # Perform Senti-word lookup
                if (pos in impt) and (word not in stopwords_defined) and \
                    ((word in self.pos['word'].tolist()) or (word in self.neg['word'].tolist()) or \
                    (word in self.ntl['word'].tolist())):
                    if pos in non_base:
                        # Find the base form of the given word
                        # i.e. -> going (as a verb) -> go for verb and nones only
                        word = wnl.lemmatize(word, pos_short(pos))
                    if (word in self.pos['word'].tolist()) and \
                        (word in self.neg['word'].tolist()):
                        if min(self.pos.loc[self.pos.word == word,'layer']) >  \
                            min(self.neg.loc[self.neg.word == word,'layer']):
                            score = -1
                        else:
                            score = 1
                    elif word in self.pos['word'].tolist():
                        score = 1
                    elif word in self.neg['word'].tolist():
                        score = -1
                    else:
                        score = 0

                    # Handle negation within neighborhood

                    if (len(set(self.intensifier.keys()).intersection(set(neighborhood)))) != 0:
                        for word in neighborhood:
                            scalar = 0
                            if word in self.intensifier.keys():
                                scalar = self.intensifier[word]
                            intensifier = intensifier * (1 + scalar)

                    if (len(negations.intersection(set(neighborhood))) == 1) & (score != 0):
                        score = score * (-1)
                    elif (len(negations.intersection(set(neighborhood))) == 1) & (score == 0):
                        score = -1
                    score = score * intensifier
                    scores.append(score)

            except: #Exception as e
                #print('There was an error: ' + str(e))
                pass
            index += 1

        emo_score = 0

        for k,v in self.emoticon.items():
            if k in sentence:
                emo_score += v

        if len(scores) > 0:
            return sum(scores) + emo_score
        else:
            return emo_score


    def weighted_score(self, business_name, reviews):
        # Compute a weighted score for each review based on the existence of NNP
        sNLP= StanfordNLP()
        score_review = 0
        # weight for each sentence if plain mean is applied
        weight = 1/len(reviews)
        weight_list = [0]*len(reviews)
        non_nnp_list = []
        score_list =[]
        idx = 0
        nnp_name = ''
        if len(reviews) == 1:
            return self.score(reviews[0])
        else:
            for j in range(len(reviews)):
                sentence = reviews[j]
                score_sen = self.score(sentence)
                score_list.append(score_sen)
                tag_pair = sNLP.pos(sNLP.to_truecase(sentence))
                word_list = [pair[0] for pair in tag_pair]
                tag_list = [pair[1] for pair in tag_pair]
                if ('NNP' in tag_list) | ('NNPS' in tag_list):
                    idx = 0
                    nnp_name = ''
                    if 'NNP' in tag_list:
                        idx = tag_list.index('NNP')
                    elif 'NNPS' in tag_list:
                        idx = tag_list.index('NNPS')

                    nnp_name = word_list[idx]
                    if nnp_name == business_name:
                        weight_multiplier = 0.5
                        weight_list[j] = weight_multiplier*weight
                    else:
                        non_nnp_list.append(j)
                else:
                    non_nnp_list.append(j)

                if len(non_nnp_list) > 0:
                    weight_non_nnp = (1 - sum(weight_list))/len(non_nnp_list)
                else:
                    weight_non_nnp = 0
            for k in non_nnp_list:
                weight_list[k] = weight_non_nnp

            return sum(x*y for x,y in zip(weight_list, score_list))

    def is_multiword(self, words):
        '''Check if a group of words is indeed a multiword expression'''
        joined = '-'.join(words)
        if (joined in self.pos['word'].tolist()) or \
            (joined in self.neg['word'].tolist()) or \
            (joined in self.ntl['word'].tolist()) or
            (joined in self.intensifier.keys()):
            return True
        else:
            return False

    def find_neighbour(self, token, pos, neighbour=5):
        read_before, read_after = [True, True]
        tokens_b = []
        tokens_a = []
        token_1_b, token_2_b,token_3_b,token_4_b = ['','','','']
        token_neigh = []
        token_1_a, token_2_a,token_3_a,token_4_a = ['','','','']
        for i in range(neighbour):
            k = i+1
            pos_before = pos - k
            pos_after = pos + k
            if read_before:
                if k == 1:
                    token_1_b = token[pos_before]
                    tokens_b = [token_1_b]
                elif k == 2:
                    token_2_b = token[pos_before] + ' ' + token_1_b
                    token_1_b = token[pos_before]
                    tokens_b = [token_2_b, token_1_b]
                elif k == 3:
                    token_3_b = token[pos_before] +  ' ' + token_2_b
                    token_2_b = token[pos_before] + ' ' + token_1_b
                    token_1_b = token[pos_before]
                    tokens_b = [token_3_b, token_2_b, token_1_b]
                else:
                    token_4_b = token[pos_before] + ' ' + token_3_b
                    token_3_b = token[pos_before] + ' ' + token_2_b
                    token_2_b = token[pos_before] + ' ' + token_1_b
                    token_1_b = token[pos_before]
                    tokens_b = [token_4_b, token_3_b, token_2_b, token_1_b]
                if any(x in tokens_b for x in self.conjunction_word) or (pos_before < 0):
                    read_before = False
                else:
                    token_neigh.extend(tokens_b)
            if read_after:
                if k == 1:
                    token_1_a = token[pos_after]
                    tokens_a = [token_1_a]
                elif k == 2:
                    token_2_a = token_1_a + " " + token[pos_after]
                    token_1_a = token[pos_after]
                    tokens_a = [token_2_a, token_1_a]
                elif k == 3:
                    token_3_a = token_2_a + " " + token[pos_after]
                    token_2_a = token_1_a + " " + token[pos_after]
                    token_1_a = token[pos_after]
                    tokens_a = [token_3_a, token_2_a, token_1_a]
                else:
                    token_4_a = token_3_a + " " + token[pos_after]
                    token_3_a = token_2_a + " " + token[pos_after]
                    token_2_a = token_1_a + " " + token[pos_after]
                    token_1_a = token[pos_after]
                    tokens_a = [token_4_a, token_3_a, token_2_a, token_1_a]
                if any(x in tokens_a for x in self.conjunction_word):
                    read_after = False
                elif pos_after >= len(token)-1:
                    read_after = False
                    token_neigh.extend(tokens_a)
                else:
                    token_neigh.extend(tokens_a)
        return token_neigh



sentimodel = yelpSentiModel()
text = "The food is five star but the service is bad."

print(sentimodel.score(text))
print(sentimodel.is_multiword(['five','star']))
print(sentimodel.is_multiword(['stand','up']))

name = 'Smashburger'
test_review = ['Bbq, bacon burger is awesome!', 'Salads are great for 2.', \
'Been coming here since it opened a few years ago for take out and dine in.', \
'Staff has always been friendly and courteous.', \
"3 rating because we love Smash fries but over the years, it's gotten more and more greasy."]
print('final score: ' + str(sentimodel.weighted_score(name,test_review)))

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
                        'other', 'some', 'such'])
transitional_word = set(["in the first place", "moreover", "as well as", "and", "also",
                        "in addition", "then", "likewise", "first", "second", "third",
                        "furthermore", "additionally", "but", "although", "in contrast",
                        "instead", "unlike", "or", "despite", "yet", "conversely",
                        "on the contrary", "while", "otherwise", "at the same time",
                        "however", "in spite of", "besides", "besides", "rather",
                        "nevertheless", "even though", "when", "so that", "whenever",
                        "while", ""])

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

class yelpSentiModel(object):
    '''A class defined obtain sentiment score based on analyser'''

    def __init__(self, filename='SentiForest.pkl',inten_file = 'intensifier_dic.txt', \
    emo_file = 'AFINN_emoticon.txt'):
        flatten_dict = flatten_tree(filename)
        # parse file and build sentiwordnet dicts
        self.intensifier = {}
        self.emoticon = {}
        self.pos = flatten_dict[flatten_dict['sentiment'] = 'Positive']
        self.neg = flatten_dict[flatten_dict['sentiment'] = 'Negative']
        self.ntl = flatten_dict[flatten_dict['sentiment'] = 'Neutral']
        self.build_emoticondict(emo_file)
        self.build_intensifier(inten_file)

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
            scalar = 0
            pos = el[1]
            try:
                word = re.match('(\w+)', el[0]).group(0).lower()
                neighborhood = self.find_neighbour(sentence_token,word,index,7)

                # look for trailing multiword expressions
                word_minus_one = sentence_token[index-1:index+1]
                word_minus_two = sentence_token[index-2:index+1]
                word_minus_three = sentence_token[index-3:index+1]
                word_minus_four = sentence_token[index-4:index+1]

                # if multiword expression, fold to one expression
                # Check multi-Senti words
                if(self.is_multiword(word_minus_three)):
                    if len(scores) > 2:
                        scores.pop()
                        scores.pop()
                    if len(neighborhood) > 2:
                        neighborhood.pop()
                        neighborhood.pop()
                    word = '-'.join(word_minus_three)
                    pos = 'unknown'
                elif(self.is_multiword(word_minus_two)):
                    if len(scores) > 1:
                        scores.pop()
                        scores.pop()
                    if len(neighborhood) > 1:
                        neighborhood.pop()
                        neighborhood.pop()
                    word = '-'.join(word_minus_two)
                    pos = 'unknown'
                elif(self.is_multiword(word_minus_one)):
                    if len(scores) > 0:
                        scores.pop()
                    if len(neighborhood) > 0:
                        neighborhood.pop()
                    word = '-'.join(word_minus_one)
                    pos = 'unknown'

                # Check if it's intensifier or not
                if word in self.intensifier.keys():
                    scalar = self.intensifier[word]
                elif '-'.joined(word_minus_one) in self.intensifier.keys():
                    intensify_word = '-'.joined(word_minus_one)
                    scalar = self.intensifier[intensify_word]
                elif '-'.joined(word_minus_two) in self.intensifier.keys():
                    intensify_word = '-'.joined(word_minus_two)
                    scalar = self.intensifier[intensify_word]
                elif '-'.joined(word_minus_three) in self.intensifier.keys():
                    intensify_word = '-'.joined(word_minus_three)
                    scalar = self.intensifier[intensify_word]
                elif '-'.joined(word_minus_four) in self.intensifier.keys():
                    intensify_word = '-'.joined(word_minus_four)
                    scalar = self.intensifier[intensify_word]
                intensifier = intensifier * (1 + scalar)

                # Perform Senti-word lookup
                if (pos in impt) and (word not in stopwords_defined):
                    if pos in non_base:
                        # Find the base form of the given word
                        # i.e. -> going (as a verb) -> go for verb and nones only
                        word = wnl.lemmatize(word, self.pos_short(pos))
                    if (word in self.pos['word']) and (word in self.neg['word']):
                        if min(self.pos.get_loc(word)['layer'] >  \
                            min(self.pos.get_loc(word)['layer']):
                            score = -1
                        else:
                            score = 1
                    elif word in self.pos['word']:
                        score = 1
                    elif word in self.neg['word']:
                        score = -1
                    else:
                        score = 0
                    # Handle negation within neighborhood
                    if (len(negations.intersection(set(neighborhood))) == 1) & (score != 0):
                        score = score * (-1)
                    elif (len(negations.intersection(set(neighborhood))) == 1) & (score == 0):
                        score = -1
                    elif (len(negations.intersection(set(neighborhood))) == 2) & (score != 0):
                        score = score
                    print(score)
                    scores.append(score)

            except:
                pass
            index += 1

        emo_score = 0
        for k,v in self.emoticon.items():
            if k in sentence:
                emo_score += v

        if len(scores) > 0:
            return intensifier * sum(scores) + emo_score
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
        weight_non_nnp = (1 - sum(weight_list))/len(non_nnp_list)
        for k in non_nnp_list:
            weight_list[k] = weight_non_nnp
        return sum(x*y for x,y in zip(weight_list, score_list))

    def is_multiword(self, words):
        '''Check if a group of words is indeed a multiword expression'''
        joined = '-'.join(words)
        if joined in self.pos['word']:
            return 'Positive'
        elif joined in self.neg['word']:
            return 'Negative'
        elif joined in self.ntl['word']:
            return 'Neutral'
        else:
            return 'Not a multiword'


    def find_neighbour(self, token, word, pos, neighbour):
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
                if any(x in tokens_b for x in transitional_word) or (pos_before < 0):
                    read_before = False
                else:
                    token_neigh.append(token[pos_before])
            if read_after:
                if k == 1:
                    token_1_a = token[pos_after]
                    tokens_a = [token_1_a]
                elif k == 2:
                    token_2_a = token[pos_after] + " " + token_1_a
                    token_1_a = token[pos_after]
                    tokens_a = [token_2_a, token_1_a]
                elif k == 3:
                    token_3_a = token[pos_after] + " " + token_2_a
                    token_2_a = token[pos_after] + " " + token_1_a
                    token_1_a = token[pos_after]
                    tokens_a = [token_3_a, token_2_a, token_1_a]
                else:
                    token_4_a = token[pos_after] + " " + token_3_a
                    token_3_a = token[pos_after] + " " + token_2_a
                    token_2_a = token[pos_after] + " " + token_1_a
                    token_1_a = token[pos_after]
                    tokens_a = [token_4_a, token_3_a, token_2_a, token_1_a]
                if any(x in tokens_a for x in transitional_word):
                    read_after = False
                elif pos_after >= len(token)-1:
                    read_after = False
                    token_neigh.append(token[pos_after])
                else:
                    token_neigh.append(token[pos_after])
        return token_neigh



sentimodel = yelpSentiModel()
text = "This place is very good. "
print(sentimodel.score(text))

'''
name = 'Smashburger'
test_review = ['Bbq, bacon burger is awesome!', 'Salads are great for 2.', \
'Been coming here since it opened a few years ago for take out and dine in.', \
'Staff has always been friendly and courteous.', \
"3 rating because we love Smash fries but over the years, it's gotten more and more greasy."]
print(sentimodel.weighted_score(name,test_review))
'''

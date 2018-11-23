# Establish a Senti Class to parse info from SentiWordNet and compute score
# for each given sentences

import nltk
import numpy
import re
from stanfordnlp_class import StanfordNLP

class baseSentiModel(object):
    '''A class defined obtain sentiment score based on analyser'''

    def __init__(self, filename='SentiWordNet.txt', weighting = 'geometric'):
        if weighting not in ('geometric', 'harmonic', 'average'):
            raise ValueError(
                'Allowed weighting options are geometric, harmonic, average')
        # parse file and build sentiwordnet dicts
        self.swn_pos = {'a': {}, 'v': {}, 'r': {}, 'n': {}}
        self.swn_all = {}
        self.build_sentidict(filename, weighting)

    def average(self, score_list):
        """Get arithmetic average of scores."""
        if(score_list):
            return sum(score_list) / float(len(score_list))
        else:
            return 0

    def geometric_weighted(self, score_list):
        """"Get geometric weighted sum of scores."""
        weighted_sum = 0
        num = 1
        for el in score_list:
            weighted_sum += (el * (1 / float(2**num)))
            num += 1
        return weighted_sum

    # another possible weighting instead of average
    def harmonic_weighted(self, score_list):
        """Get harmonic weighted sum of scores."""
        weighted_sum = 0
        num = 2
        for el in score_list:
            weighted_sum += (el * (1 / float(num)))
            num += 1
        return weighted_sum

    def build_sentidict(self, filename, weighting):
        '''Extract sentiment score from file'''

        records = [line.strip().split('\t') for line in open(filename)]
        for rec in records:
            words = rec[4].split()
            pos = rec[0]

            for aword in words:
                word = aword.split('#')[0]
                sense_num = int(aword.split('#')[1])

                if word not in self.swn_pos[pos]:
                    self.swn_pos[pos][word] = {}
                self.swn_pos[pos][word][sense_num] = 5*(float(rec[2]) - float(rec[3]))
                if word not in self.swn_all:
                    self.swn_all[word] = {}
                self.swn_all[word][sense_num] = 5*(float(rec[2]) - float(rec[3]))

                # Sort senti scores based on sense_num
        for pos_key in self.swn_pos.keys():
            for word_key in self.swn_pos[pos_key].keys():
                sorted_score = [self.swn_pos[pos_key][word_key][k] for k in
                                sorted(self.swn_pos[pos_key][word_key].keys())]
                if weighting == 'average':
                    self.swn_pos[pos_key][word_key] = self.average(sorted_score)
                if weighting == 'geometric':
                    self.swn_pos[pos_key][word_key] = self.geometric_weighted(sorted_score)
                if weighting == 'harmonic':
                    self.swn_pos[pos_key][word_key] = self.harmonic_weighted(sorted_score)

        for word_key in self.swn_all.keys():
            sorted_score = [self.swn_all[word_key][k] for k in
                            sorted(self.swn_all[word_key].keys())]
            if weighting == 'average':
                self.swn_all[word_key] = self.average(sorted_score)
            if weighting == 'geometric':
                self.swn_all[word_key] = self.geometric_weighted(sorted_score)
            if weighting == 'harmonic':
                self.swn_all[word_key] = self.harmonic_weighted(sorted_score)


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

    def score_word(self, word, pos):
        # Assign a score to the word based on its pos
        try:
            return self.swn_pos[pos][word]
        except KeyError:
            try:
                return self.swm_all[word]
            except KeyError:
                return 0

    def score(self, sentence):
        '''Sentiment score of a given sentence, assuming sentence has been tokenzied'''
        # init sentiwordnet lookup/scoring tools
        impt = set(['NNS', 'NN', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
                    'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN',
                    'VBP', 'VBZ', 'unknown'])
        non_base = set(['VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNS', 'NNPS'])
        negations = set(['not', 'n\'t', 'less', 'no', 'never',
                         'nothing', 'nowhere', 'hardly', 'barely',
                         'scarcely', 'nobody', 'none', 'dont',
                         'cant', 'couldnt',
                         'wont','wouldnt', 'doesnt'])
        amplifier = set(['very', 'ever', 'always', 'super', '!', 'fucking',
                         'damn', 'ridiculously', 'most', 'even'])
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
                             'more', 'other', 'some', 'such'])
        amp = 2
        amp_exist = False
        wnl = nltk.WordNetLemmatizer()
        sNLP= StanfordNLP()
        sentence_token = nltk.word_tokenize(sentence)
        tagger = sNLP.pos(sNLP.to_truecase(sentence))

        # Assume that sentence is already in tokenized form
        scores = []

        index = 0
        for el in tagger:
            pos = el[1]
            try:
                word = re.match('(\w+)', el[0]).group(0).lower()
                start = index - 10
                if start < 0:
                    start = 0
                neighborhood = sentence_token[start:index]
                # look for trailing multiword expressions
                word_minus_one = sentence_token[index-1:index+1]
                word_minus_two = sentence_token[index-2:index+1]

                # if multiword expression, fold to one expression
                if(self.is_multiword(word_minus_two)):
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
                # perform lookup
                if (pos in impt) and (word not in stopwords_defined):
                    if pos in non_base:
                        # Find the base form of the given word
                        # i.e. -> going (as a verb) -> go for verb and nones only
                        word = wnl.lemmatize(word, self.pos_short(pos))
                    score = self.score_word(word, self.pos_short(pos))
                    if len(negations.intersection(set(neighborhood))) == 1:
                        score = score + 4*((-1)**(sign + 1))
                    elif len(negations.intersection(set(neighborhood))) == 2:
                        score = score - 2*((-1)**(sign))
                    if len(amplifier.intersection(set(neighborhood))) > 0:
                        amp_exist = True
                    scores.append(score)

            except:
                pass
            index += 1
        if len(scores) > 0:
            if amp_exist == True:
                amp = 2
            else:
                amp = 1
            return amp * sum(scores)
        else:
            return 0

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
            score_list.append(self.score(sentence))
            tag_pair = sNLP.pos(sNLP.to_truecase(sentence))
            word_list = [pair[0] for pair in tag_pair]
            tag_list = [pair[1] for pair in tag_pair]
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
        weight_non_nnp = (1 - sum(weight_list))/len(non_nnp_list)
        for k in non_nnp_list:
            weight_list[k] = weight_non_nnp
        return sum(x*y for x,y in zip(weight_list, score_list))

    def is_multiword(self, words):
        '''Check if a group of words is indeed a multiword expression'''
        joined = '_'.join(words)
        return joined in self.swn_all

sentimodel = baseSentiModel()
text = "This was not the best restaurant I've been but it was ok. "
print(sentimodel.score(text))

name = 'Smashburger'
test_review = ['Bbq, bacon burger is awesome!', 'Salads are great for 2.', \
'Been coming here since it opened a few years ago for take out and dine in.', \
'Staff has always been friendly and courteous.', \
"3 rating because we love Smash fries but over the years, it's gotten more and more greasy."]
print(sentimodel.weighted_score(name,test_review))

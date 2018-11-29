from nltk.corpus import wordnet as wn

synonyms = []
antonyms = []

def get_syn(word):
    synonyms = []
    antonyms = []
    for syn in wn.synsets(word):
#syn = wn.synsets('fast')
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
        for s in syn.similar_tos():
            synonyms.append(s.name().split('.')[0])
    return {'synonyms':list(set(synonyms)),\
            'antonyms':list(set(antonyms))}

print(get_syn('fast'))

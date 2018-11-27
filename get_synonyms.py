from nltk.corpus import wordnet as wn

synonyms = []
antonyms = []

for syn in wn.synsets('fast'):
#syn = wn.synsets('fast')
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
    for s in syn.similar_tos():
        synonyms.append(s.name().split('.')[0])


print(set(synonyms), set(antonyms))

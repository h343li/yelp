from stanfordcorenlp import StanfordCoreNLP
import logging
import json

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'truecase',
            'pipelineLanguage': 'en',
            'outputFormat': 'json',
            'truecase.overwriteText' : 'true'
        }

    def to_truecase(self, sentence):
        sentences = json.loads(self.nlp.annotate(sentence, properties=self.props))['sentences']
        normal_text = ""
        for sentence in sentences:
            tokens = sentence['tokens']
            for js in tokens:
                normal_text = normal_text + js['word'] + " "
        return normal_text

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

if __name__ == '__main__':
    sNLP = StanfordNLP()
    text = "john donk works POI Jones wants meet Xyz Corp measuring POI short term performance metrics. "
#    print ("Annotate:{}".format(sNLP.annotate(text)))
#    print ("Tokens:{}".format(sNLP.word_tokenize(text)))
#    print ("NER:{}".format(sNLP.ner(text)))
#    print ("Parse:{}".format(sNLP.parse(text)))
#    print ("Dep Parse:{}".format(sNLP.dependency_parse(text)))
    print ("POS:{}".format(sNLP.pos(sNLP.to_truecase(text))))

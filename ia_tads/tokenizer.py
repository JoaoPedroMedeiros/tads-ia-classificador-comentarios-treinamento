import nltk
from nltk.stem import RSLPStemmer

class JuizTokenizer:
    def __call__(self, text):
        tokens = self._tokenize(text)
        stemmed_tokens = self._stem(tokens)
        return self._remove_stop_words(stemmed_tokens)

    @staticmethod
    def _remove_stop_words(text):
        """ Remove stop words of the text

        :param text: list of words to be checked
        :return: list of words without stop words
        """
        stopwords = nltk.corpus.stopwords.words('portuguese')
        phrase = []
        for word in text:
            if word not in stopwords:
                phrase.append(word)
        return phrase

    @staticmethod
    def _stem(text):
        """ Convert words to it's stem

        :param text: list of words
        :return: list of stemmed words
        """
        stemmer = RSLPStemmer()
        phrase = []
        for word in text:
            phrase.append(stemmer.stem(word.lower()))
        return phrase

    @staticmethod
    def _tokenize(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        return text

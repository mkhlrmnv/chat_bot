import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def stem_list(sentence):
    result = []
    for w in sentence:
        result.append(stem(w))
    return result


def bag_of_words(sentence, words):
    sentence_words = stem_list(sentence)

    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag



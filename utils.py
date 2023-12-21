import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# nltk.download('punkt')
stemmer = PorterStemmer()

ignore_chars = ['?', '!', ',', '.', '"', "'"]

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
            bag[idx] = 1.0

    return bag


def ignore_symbols(list):
    result = []
    for w in list:
        if w not in ignore_chars:
            result.append(w)
    return result

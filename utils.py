import nltk
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(sentence, words):
    pass

# Testing functions
a = "How long does shipping take"
a = tokenize(a)
print(a)

b = ["organize", "organization", "organizing"]
stemmed = [stem(w) for w in b]
print(stemmed)


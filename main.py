import spacy
nlp = spacy.load('en_core_web_sm')

text = "The quick brown fox jumped over the lazy dog."

parsed = nlp(text)

for token in parsed:
  print(token.text, token.tag_, token.head.text, token.dep_)

#spacy.displacy.serve(parsed, style='dep')

# Text Categorization
# https://www.nltk.org/book/ch06.html

def gender_features(name):
    return {
      'last_letter': name[-1]
    }

import nltk
nltk.download('names')

from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
import random
random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
#classifier = nltk.NaiveBayesClassifier.train(train_set)

#print(nltk.classify.accuracy(classifier, test_set))

#classifier.show_most_informative_features(10)

#maxEnt = nltk.MaxentClassifier.train(train_set)

#print(nltk.classify.accuracy(classifier, test_set))

#classifier.show_most_informative_features(10)

# Stemming

nltk.download('wordnet')
nltk.download('gutenberg')

words = nltk.corpus.gutenberg.words('austen-sense.txt')

stemmer = nltk.stem.SnowballStemmer('english')

for word in words[60:80]:
  print(stemmer.stem(word) + ' is the stem of ' + word)


import nltk
from nltk.corpus import brown

nltk.download('brown')

genre_word = [
  (genre, word)
  for genre in brown.categories()
  for word in brown.words(categories=genre)
]

import random
random.shuffle(genre_word)

featuresets = [({word: 1}, genre) for (genre, word) in genre_word]

train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.show_most_informative_features(100)

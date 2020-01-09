#
#  Borrowed from https://github.com/cair/fast-tsetlin-machine-in-cuda-with-imdb-demo/blob/master/produce_dataset.py.
#  Copyright (c) 2019 Ole-Christoffer Granmo
#  This file is under the MIT license: https://github.com/cair/fast-tsetlin-machine-in-cuda-with-imdb-demo/blob/master/LICENSE.
#

import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

NUM_WORDS = 5000
MAX_NGRAM = 3
INDEX_FROM = 2
FEATURES = 10000

np.load.__defaults__ = (None, True, True, 'ASCII')
train, test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
np.load.__defaults__ = (None, False, True, 'ASCII')

train_x, train_y = train
test_x, test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value: key for key, value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
    terms = []
    for word_id in train_x[i]:
        terms.append(id_to_word[word_id])

    for N in range(1, MAX_NGRAM + 1):
        grams = [terms[j:j + N] for j in range(len(terms) - N + 1)]
        for gram in grams:
            phrase = " ".join(gram)

            if phrase in vocabulary:
                vocabulary[phrase] += 1
            else:
                vocabulary[phrase] = 1

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
    if vocabulary[phrase] < 10:
        continue

    phrase_bit_nr[phrase] = bit_nr
    bit_nr_phrase[bit_nr] = phrase
    bit_nr += 1

# Create bit representation

X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.int32)
y_train = np.zeros(train_y.shape[0], dtype=np.int32)
for i in range(train_y.shape[0]):
    terms = []
    for word_id in train_x[i]:
        terms.append(id_to_word[word_id])

    for N in range(1, MAX_NGRAM + 1):
        grams = [terms[j:j + N] for j in range(len(terms) - N + 1)]
        for gram in grams:
            phrase = " ".join(gram)
            if phrase in phrase_bit_nr:
                X_train[i, phrase_bit_nr[phrase]] = 1

    y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.int32)
y_test = np.zeros(test_y.shape[0], dtype=np.int32)

for i in range(test_y.shape[0]):
    terms = []
    for word_id in test_x[i]:
        terms.append(id_to_word[word_id])

    for N in range(1, MAX_NGRAM + 1):
        grams = [terms[j:j + N] for j in range(len(terms) - N + 1)]
        for gram in grams:
            phrase = " ".join(gram)
            if phrase in phrase_bit_nr:
                X_test[i, phrase_bit_nr[phrase]] = 1

    y_test[i] = test_y[i]

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

output_test = np.c_[X_test, y_test]
np.savetxt("imdb-test.data", output_test, fmt="%d")

output_train = np.c_[X_train, y_train]
np.savetxt("imdb-train.data", output_train, fmt="%d")

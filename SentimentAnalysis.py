# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:35:00 2018

@author: Vishnu
"""

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

seed = 7
np.random.seed(seed)

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

def get_lstm_model():
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = get_lstm_model()
model.compile(loss='binary_crossentropy', optimizer = 'adam',metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=128, verbose=2)
score, acc = model.evaluate(X_test, y_test, verbose=0)
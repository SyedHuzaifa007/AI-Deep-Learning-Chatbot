# Importing Modules
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # Tokenize Each Word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add Documents In The Corpus
        documents.append((w, intent['tag']))

        # Add To Our Classes List:
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
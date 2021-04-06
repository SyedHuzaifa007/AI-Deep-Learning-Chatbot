# Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import nltk
nltk.download('punkt')
nltk.download('worknet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

# Initailizing Chatbot Training
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # Take Each Word And Tokenizing It
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Adding Documents
        documents.append((w, intent['tag']))

        # Adding Classes To Our Class List
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Extracting All The Words Within "patterns"
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print(len(set(classes)))
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

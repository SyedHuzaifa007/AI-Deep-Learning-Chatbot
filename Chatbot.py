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


# Building Deep Learning Model
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # Intializing Bag Of Words
    bag = []
    # List Of Tokenized Words For Patterns
    pattern_words = doc[0]
    # Lemmatize Each Word - Create Base Word, In Attempt Related To Represent Related Words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create Our Bag Of Words Array With 1, If Word Match Found In Current Position
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is '0' For Each Tag And '1' For Current Tag (For Each Pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag, output_row)

# Shuffle Our Features And Turn Into Numpy Array
random.shuffle(training)
training  = np.array(training)
# Create Train And Test Lists: X - Patterns, Y - Intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training Data Created")
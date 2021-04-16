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
            
# Lemmatize Lower Each Word And Remove Duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort Classes
classes = sorted(list(set(classes)))
# Documents = combinations between patterns and intents
print(len(documents), 'documents')
# classes = intents
print(len(classes), 'classes', classes)
print(len(words), 'unique lemmatized words', words)
pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Creating Training And Testing Data

# Creating Our Training Data
training = []
# Create An Empty Array For Our Output
output_empty = [0] * len(classes)
# Training set, bag of words for each sentence
for doc in documents:
    # Initializing Our Bag Of Words
    bag = []
    # List Of Tokenized Words For The Pattern
    pattern_words = doc[0]
    # Lemmatized Each Word - Create Base Word, In Attempt To Represent Related Words
    patern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
# Create Our Bag Of Words Array With 1, If Word Match Found In Current Pattern
    for w in words:
        bag.append(1) if w in patern_words else bag.append(0)

        # Output Is A '0' For Each Tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

        # Shuffle Our Features And Turn Into np.array
        random.shuffle(training)
        training = np.array(training)
            
        # Create Train And Tests Lists. X - patterns, Y - intents
        train_x = list(training[:,0])
        train_y = list(training[:,1])
        print("Training Data Created")
        

train_x
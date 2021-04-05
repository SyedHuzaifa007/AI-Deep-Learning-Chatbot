# Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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

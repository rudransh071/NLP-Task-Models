from __future__ import print_function

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, TensorBoard
from bayes_opt import BayesianOptimization
import preprocessing
import train
'''Reaches 99.78% accuracy on task 'single_supporting_fact_10k' after 300 epochs on
hyperparameters estimated by BayesianOptimization. Hyperparameters that were tuned included 
dropout value in the dropout layers, the number of dropout layers and the batch size. The 
values which minimize our loss are {"dropout_val = 0.2", "dropout_layer = 1", "lstm_dimension" = 29
, "batch_size = 41"}. This model beats the model of the original author by 1.18%.
'''
model, history = train.objective(0.2, 1, 29, 41, True)
data = preprocessing.get_data()

story_maxlen = int(data['story_maxlen'])
query_maxlen = int(data['query_maxlen'])
vocab_size = int(data['vocab_size'])
inputs_train = data['inputs_train']
inputs_test = data['inputs_test']
answers_train = data['answers_train']
answers_test = data['answers_test']
queries_train = data['queries_train']
queries_test = data['queries_test']

model.load_weights('weights.hdf5')
pred_results = model.predict(([inputs_test, queries_test]))
# Display a selected test story

n = np.random.randint(0,1000)
story_list = test_stories[n][0]
story =' '.join(word for word in story_list)
print("Story is:",story)

question_list = test_stories[n][1]
ques =' '.join(word for word in question_list)
print("Question is: ",ques)

ans = test_stories[n][2]
print("Actual answer is: ", ans)

#Generate prediction from model

val_max = np.argmax(pred_results[n])

for key, val in word_idx.items():
    if val == val_max:
        k = key

print("Machine answer is: ", k)
print("I am ", pred_results[n][val_max], "certain of it")

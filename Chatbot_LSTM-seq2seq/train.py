'''Trains a memory network on the bAbI dataset.
References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
'''
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

def objective(dropout_val, dropout_layers, lstm_val, batch_size, create_save_plot = False):
    # placeholders
    input_sequence = Input((story_maxlen,))
    question = Input((query_maxlen,))

    # encoders
    # embed the input sequence into a sequence of vectors
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size,
                                  output_dim=64))
    input_encoder_m.add(Dropout(dropout_val))
    # output: (samples, story_maxlen, embedding_dim)

    # embed the input into a sequence of vectors of size query_maxlen
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size,
                                  output_dim=query_maxlen))
    input_encoder_c.add(Dropout(dropout_val))
    # output: (samples, story_maxlen, query_maxlen)

    # embed the question into a sequence of vectors
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                                   output_dim=64,
                                   input_length=query_maxlen))
    question_encoder.add(Dropout(dropout_val))
    # output: (samples, query_maxlen, embedding_dim)

    # encode input sequence and questions (which are indices)
    # to sequences of dense vectors
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    # compute a 'match' between the first input vector sequence
    # and the question vector sequence
    # shape: `(samples, story_maxlen, query_maxlen)`
    match = dot([input_encoded_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)

    # add the match matrix with the second input vector sequence
    response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
    response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

    # concatenate the match matrix with the question vector sequence
    answer = concatenate([response, question_encoded])

    # the original paper uses a matrix multiplication for this reduction step.
    # we choose to use a RNN instead.
    answer = LSTM(int(lstm_val))(answer)  # (samples, 32)

    # one regularization layer -- more would probably be needed.
    for i in range((int)(dropout_layers)):
        answer = Dropout(dropout_val)(answer)
        answer = Dropout(dropout_val)(answer)
    answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
    # we output a probability distribution over the vocabulary
    answer = Activation('softmax')(answer)

    # build the final model
    model = Model([input_sequence, question], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train
    callback = [ModelCheckpoint('weights.hdf5',
                                    save_best_only=True,
                                    save_weights_only=True, period = 10)]
    history = model.fit([inputs_train, queries_train], answers_train,
              batch_size=int(batch_size),
              epochs=300,
              validation_data=([inputs_test, queries_test], answers_test), callbacks = callback)
    print(history.history.keys())
    # summarize history for accuracy
    if create_save_plot:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy_epochs.png', dpi=200)
    return model, history

pbounds = {'dropout_val':(0.2, 0.4), 'dropout_layers':(1, 3), 'batch_size':(32, 64), 'lstm_val':(25, 40)}
optimizer1 = BayesianOptimization(f = objective, pbounds = pbounds, random_state = 1)
optimizer1.maximize(n_iter = 25, init_points = 2)
history, model = objective()



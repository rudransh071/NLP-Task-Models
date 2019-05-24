from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.layers import *
import numpy as np
import os

top_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

temp_word_to_idx = imdb.get_word_index()
idx_to_word = {}
word_to_idx = {}

for key, value in imdb.get_word_index().items():
	word_to_idx[key] = value+3

for key, value in word_to_idx.items():
	idx_to_word[value] = key

idx_to_word[0] = '<PAD>'
idx_to_word[1] = '<START>'
idx_to_word[2] = '<UNK>'

review = ""
for token in x_train[0]:
	review = review + idx_to_word[token] + " "

print("Sample Review " + review)

for i in range(x_train.shape[0]):
	x_train[i] = np.asarray(x_train[i])

for i in range(x_test.shape[0]):
	x_test[i] = np.asarray(x_test[i])

x_train_pad = pad_sequences(x_train, maxlen=600,
                            padding='pre', truncating='pre')

x_test_pad = pad_sequences(x_test, maxlen=600,
                           padding='pre', truncating='pre')

embeddings_index = {}
f = open(os.path.join('Glove_embedding/glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_to_idx) + 1, 100))
for word, i in word_to_idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

input = Input(shape = (600,))
embedding_layer = Embedding(len(word_to_idx)+1, 100, weights = [embedding_matrix], input_length = 600, 
							trainable = False)
GRU_1 = GRU(units = 16, name = "GRU_1", return_sequences = True)
GRU_2 = GRU(units = 8, name = "GRU_2", return_sequences = True)
GRU_3 = GRU(units = 4, name = "GRU_3", return_sequences = False)
dense_1 = Dense(units = 1, activation = 'sigmoid', name = "DENSE")

first_layer = embedding_layer(input)
second_layer = GRU_1(first_layer)
third_layer = GRU_2(second_layer)
fourth_layer = GRU_3(third_layer)
fifth_layer = dense_1(fourth_layer)

model = Model(input, fifth_layer)
model.compile('adam', 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train_pad, y_train, epochs = 50, batch_size = 128, verbose = 1, validation_data = (x_test_pad, y_test))

model.evaluate(x_test, y_test, verbose = 0)


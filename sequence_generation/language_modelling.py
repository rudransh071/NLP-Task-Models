from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Input, GRU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import RepeatVector
import keras.utils as ku 
from keras.datasets import imdb
import numpy as np 
import os
from keras.models import Model

num_words = 10000
tokenizer = Tokenizer(num_words = 10000)

data = open('text_generation.pdf').read()
text = data.lower().split("\n")

tokenizer.fit_on_texts(text)

input_sequences = []
for line in text:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

embeddings_index = {}
f = open(os.path.join('Glove_embedding/glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
# label = ku.to_categorical(label,len(tokenizer.word_index)+1)

input = Input(shape=(predictors.shape[1],), name = "input_layer")
embedding_layer = Embedding(len(tokenizer.word_index)+1, 100, input_length = predictors.shape[1], 
							weights = [embedding_matrix], trainable = False, name = "embedding_layer")
GRU_1 = GRU(units = 50, name = "GRU_1")
dense_1 = Dense(units = len(tokenizer.word_index)+1, activation = "softmax", name = "DENSE")

first_layer = embedding_layer(input)
second_layer = GRU_1(first_layer)
fourth_layer = dense_1(second_layer)

model2 = Model(input, fourth_layer)
print(model2.summary())
model2.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

def train(predictors, label, batch_size, epochs, model):
	for i in range(epochs):
		for j in range(0, len(predictors)-batch_size, batch_size):
			predictor_tmp = predictors[j:j+batch_size]
			len(predictor_tmp)
			label_tmp = label[j:j+batch_size]
			label_tmp = ku.to_categorical(label_tmp, len(tokenizer.word_index)+1)
			model2.fit(predictor_tmp, label_tmp, epochs = 1, batch_size = batch_size)

train(predictors, label, 1024, 3, model2)

def generate_text(seed_text, next_words, max_sequence_len):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predict_model = model2.predict(token_list, verbose=0)
		predicted = np.argmax(predict_model, axis = 1)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text


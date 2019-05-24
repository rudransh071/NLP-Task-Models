from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from pickle import dump, load
from keras.preprocessing.text import Tokenizer
import keras.utils as ku
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import math

batch_size = 64
epochs = 110
latent_dim = 256
num_samples = 10000
num_words = 2000

tokenizer = Tokenizer(num_words=num_words)

stories = load(open('review_dataset.pkl', 'rb'))
print('Loaded Stories %d' % len(stories))
print(type(stories))

input_texts = []
target_texts = []

for story in stories:
	input_texts.append(story['story'])
	target_texts.append(story['highlights'])

input_texts = input_texts[:50000]
target_texts = target_texts[:10000]

tmp = input_texts+target_texts
tokenizer.fit_on_texts(tmp)

for i in range(len(input_texts)):
	tokens = input_texts[i].split()
	input_texts[i] = tokens
	input_texts[i] = tokenizer.texts_to_sequences(input_texts[i])
	

for i in range(len(target_texts)):
	tokens = target_texts[i].split()
	target_texts[i] = tokens
	target_texts[i] = tokenizer.texts_to_sequences(target_texts[i])

for i in range(len(input_texts)):
	tmpvar = []
	for j in range(len(input_texts[i])):
		if len(input_texts[i][j]) > 0:
			tmpvar.append(input_texts[i][j][0])
	input_texts[i] = tmpvar

for i in range(len(target_texts)):
	tmpvar = []
	for j in range(len(target_texts[i])):
		if len(target_texts[i][j]) > 0:
			tmpvar.append(target_texts[i][j][0])
	target_texts[i] = tmpvar

input_len_array_sizes = np.zeros((len(input_texts), ))
target_len_array_sizes = np.zeros((len(target_texts), ))

for i in range(len(input_texts)):
	input_len_array_sizes[i] = (len(input_texts[i]))

for i in range(len(target_texts)):
	target_len_array_sizes[i] = (len(target_texts[i]))

max_len_input = math.ceil(input_len_array_sizes.mean() + 2*input_len_array_sizes.std())
max_len_target = math.ceil(target_len_array_sizes.mean() + 2*target_len_array_sizes.std())

x_train = pad_sequences(input_texts, maxlen = max_len_input, padding = 'post')
y_train = pad_sequences(target_texts, maxlen = max_len_target, padding = 'post')

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	embedding = Embedding(input_dim = vocab_size, output_dim = embedding_size)
	embedded_encoder_inputs = encoder_embedding(encoder_inputs)
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(embedded_encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	embedded_decoder_inputs = embedding(decoder_inputs)
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(embedded_decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]


# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# define model
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
# train model
train.fit([X1, X2], y, epochs=1)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))


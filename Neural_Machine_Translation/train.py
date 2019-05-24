import tensorflow.keras as keras

from tensorflow.python.keras.utils import to_categorical
import numpy as np
import os, sys

project_path = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:5])
if project_path not in sys.path:
    sys.path.append(project_path)

from utils.data_helper import read_data, sents2sequences
from model import define_nmt
from utils.model_helper import plot_attention_weights



def get_data(train_size, random_seed=100):

    """ Getting randomly shuffled training / testing data """
    eng_text = []
    ger_text = []

    with open('europarl-v7.de-en.en') as f:
        eng_text = [next(f) for x in range(train_size)]
    f.close()
    with open('europarl-v7.de-en.de') as f:
        ger_text = [next(f) for x in range(train_size)]
    f.close()
    for i in range(len(eng_text)):
        eng_text[i] = eng_text[i].strip()

    for i in range(len(ger_text)):
        ger_text[i] = ger_text[i].strip()
        
    ger_text = ['sos ' + sent[:-1] + 'eos .'  if sent.endswith('.') else 'sos ' + sent + ' eos .' for sent in ger_text]

    np.random.seed(random_seed)
    inds = np.arange(len(eng_text))
    np.random.shuffle(inds)

    train_inds = inds[:train_size]
    eng_text = [eng_text[ti] for ti in train_inds]
    ger_text = [ger_text[ti] for ti in train_inds]

    return eng_text, ger_text


def preprocess_data(en_tokenizer, fr_tokenizer, en_text, fr_text, en_timesteps, fr_timesteps):
    """ Preprocessing data and getting a sequence of word indices """

    en_seq = sents2sequences(en_tokenizer, en_text, reverse=False, padding_type='pre', pad_length=en_timesteps)
    fr_seq = sents2sequences(fr_tokenizer, fr_text, pad_length=fr_timesteps)
    return en_seq, fr_seq


def train(full_model, eng_seq, ger_seq, batch_size, n_epochs=10):
    """ Training the model """

    for ep in range(n_epochs):
        losses = []
        for bi in range(0, eng_seq.shape[0] - batch_size, batch_size):

            encoder_input = eng_seq[bi:bi+batch_size]
            decoder_input = ger_seq[bi:bi+batch_size, :-1]
            decoder_output = to_categorical(ger_seq[bi:bi+batch_size, 1:], num_classes = ger_vocab_size)

            full_model.fit([encoder_input, decoder_input], decoder_output)

            # l = full_model.evaluate([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :],
            #                         batch_size=batch_size, verbose=0)

    #         losses.append(l)
    # return losses


def infer_nmt(encoder_model, decoder_model, test_en_seq, en_vsize, fr_vsize):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_en_seq: sequence of word ids
    :param en_vsize: int
    :param fr_vsize: int
    :return:
    """

    test_fr_seq = sents2sequences(fr_tokenizer, ['sos'], fr_vsize)
    test_en_onehot_seq = to_categorical(test_en_seq, num_classes=en_vsize)
    test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

    enc_outs, enc_fwd_state, enc_back_state = encoder_model.predict(test_en_onehot_seq)
    dec_fwd_state, dec_back_state = enc_fwd_state, enc_back_state
    attention_weights = []
    fr_text = ''
    for i in range(20):

        dec_out, attention, dec_fwd_state, dec_back_state = decoder_model.predict(
            [enc_outs, dec_fwd_state, dec_back_state, test_fr_onehot_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]

        if dec_ind == 0:
            break
        test_fr_seq = sents2sequences(fr_tokenizer, [fr_index2word[dec_ind]], fr_vsize)
        test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

        attention_weights.append((dec_ind, attention))
        fr_text += fr_index2word[dec_ind] + ' '

    return fr_text, attention_weights


if __name__ == '__main__':
    debug = False
    """ Hyperparameters """
    batch_size = 8
    hidden_size = 48
    eng_timesteps, ger_timesteps = 20, 20
    train_size = 5000 if not debug else 10000
    filename = ''
    eng_text, ger_text = get_data(train_size = train_size)

    """ Defining tokenizers """
    tokenizer_eng = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    tokenizer_eng.fit_on_texts(eng_text)

    tokenizer_ger = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    tokenizer_ger.fit_on_texts(ger_text)

    """ Getting preprocessed data """
    eng_seq, ger_seq = preprocess_data(tokenizer_eng, tokenizer_ger, eng_text, ger_text, eng_timesteps, ger_timesteps)

    eng_vocab_size = max(tokenizer_eng.index_word.keys()) + 1
    ger_vocab_size = max(tokenizer_ger.index_word.keys()) + 1

    """ Defining the full model """
    full_model, infer_enc_model, infer_dec_model = define_nmt(
        hidden_size=hidden_size, batch_size=batch_size,
        eng_timesteps=eng_timesteps, ger_timesteps=ger_timesteps,
        eng_vocab_size=eng_vocab_size, ger_vocab_size=ger_vocab_size)

    n_epochs = 1 if not debug else 3
    train(full_model, eng_seq, ger_seq, batch_size, n_epochs)

    """ Save model """
    if not os.path.exists(os.path.join('..', 'h5.models')):
        os.mkdir(os.path.join('..', 'h5.models'))
    full_model.save(os.path.join('..', 'h5.models', 'nmt.h5'))

    """ Index2word """
    en_index2word = dict(zip(en_tokenizer.word_index.values(), en_tokenizer.word_index.keys()))
    fr_index2word = dict(zip(fr_tokenizer.word_index.values(), fr_tokenizer.word_index.keys()))

    """ Inferring with trained model """
    test_en = ts_en_text[0]
    logger.info('Translating: {}'.format(test_en))

    test_en_seq = sents2sequences(en_tokenizer, [test_en], pad_length=en_timesteps)
    test_fr, attn_weights = infer_nmt(
        encoder_model = infer_enc_model, decoder_model = infer_dec_model,
        test_en_seq = test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize)
    logger.info('\tFrench: {}'.format(test_fr))

    """ Attention plotting """
    plot_attention_weights(test_en_seq, attn_weights, en_index2word, fr_index2word)
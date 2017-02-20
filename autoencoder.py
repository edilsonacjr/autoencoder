
import numpy as np

from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM, RepeatVector, Dense, Dropout
from keras.models import Model

from gensim.models import Word2Vec

MAX_SEQUENCE_LENGTH = 7
EMBEDDING_DIM = 300
HIDDEN_DIM = 300

def main():


    # all_sentences must receive a list of senteces/strings that will be used on training
    all_sentences = ['list of sentences/string', 'list of sentences/string']

    tokenizer = Tokenizer() # nb_words=MAX_NB_WORDS
    tokenizer.fit_on_texts(all_sentences)
    sequences = tokenizer.texts_to_sequences(all_sentences)
    #print(sequences)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    #print(word_index.items())

    x_train = pad_sequences(sequences)
    y_train = tokenizer.texts_to_matrix(all_sentences, mode='binary')

    print('Shape of data tensor:', x_train.shape)

    # Loading Word2Vec
    model = Word2Vec.load_word2vec_format('/home/edilson/GoogleNews-vectors-negative300.bin', binary=True)

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in model:
            embedding_matrix[i] = model[word]
        else:
            embedding_matrix[i] = np.random.rand(1, EMBEDDING_DIM)[0]

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    embedded_sequences = embedding_layer(inputs)
    encoded = LSTM(HIDDEN_DIM)(embedded_sequences)

    decoded = RepeatVector(MAX_SEQUENCE_LENGTH)(encoded)
    decoded = LSTM(len(word_index))(decoded)

    #decoded = Dropout(0.5)(decoded)
    decoded = Dense(y_train.shape[1], activation='softmax')(decoded)

    sequence_autoencoder = Model(inputs, decoded)

    encoder = Model(inputs, encoded)

    sequence_autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    sequence_autoencoder.fit(x_train, y_train,
                             nb_epoch=10,
                             batch_size=32,
                             shuffle=True)
                             #metrics=['acc'])
                             #validation_data=(x_train, y_train))

if __name__ == '__main__':
    main()
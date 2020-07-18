import os
import numpy as np
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

def main():
    parse_aclimdb()

def parse_aclimdb(trainORtest_dir=None, max_words=10000, maxlen=100):
    '''
    :param trainORtest_dir: dir of aclimdb train or test
    :param max_words: the most common `max_words-1` words will be kept.
    :param maxlen: max length of a comment
    :return: array of data (data_num, maxlen) and labels (data_num,)
    '''
    if trainORtest_dir is None:
        imdb_dir = r'E:/study/机器学习/dataset/imdb/aclImdb/aclImdb'
        trainORtest_dir = os.path.join(imdb_dir, 'train')
    ## load text from file
    texts  = []
    labels = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(trainORtest_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                with open(os.path.join(dir_name, fname),encoding='utf-8') as f:
                    texts.append(f.read())#str
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    # print(len(texts))
    tokenizer = Tokenizer(num_words=max_words)  # Only the most common `num_words-1` words will be kept in each sequence
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index #dict {word: i}
    print('Found %s unique tokens.' % len(word_index)) #88582

    sequences = tokenizer.texts_to_sequences(texts) #Transforms each text in texts to a sequence of integers, return a list of sequences
    print('Max index in sequences:', max([max(x) for x in sequences])) #9999
    data = pad_sequences(sequences, maxlen=maxlen) # 将列表转化为(samples, word_indices)
    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    return data, labels, word_index

def split_val(data, labels, train_samples=200, valid_samples=10000):
    ## shuffle data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    x_train = data[:train_samples]
    y_train = labels[:train_samples]
    x_val = data[train_samples: train_samples + valid_samples]
    y_val = labels[train_samples: train_samples + valid_samples]\

    return x_train, y_train, x_val, y_val

def parse_glove(max_words, word_index, glove_dir=None, dim=100):
    '''
    :param max_words:
    :param word_index: word_index包含了所有的词，index应该是按照词频排序，词频越高index越小
    :param glove_dir:
    :param dim:
    :return:
    '''
    if glove_dir is None:
        glove_dir = 'E:/study/机器学习/dataset/glove.6B'
    embeddings_index = {}#{word: vec}
    with open(os.path.join(glove_dir, 'glove.6B.%sd.txt'%dim)) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors in glove.6B.%sd.txt'%(len(embeddings_index), dim))

    embedding_matrix = np.zeros((max_words, dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector#注意这里的 index 和 word_index 对应
    return embedding_matrix

def model(max_words, embedding_dim, maxlen, embedding_matrix, x_train, y_train, x_val, y_val):
    model = Sequential()

    model.add(layers.Embedding(input_dim=max_words,
                               output_dim=embedding_dim,
                               input_length=maxlen))#x_train 中每个序列的20个元素是单词索引，这与embedding_matrix索引对应
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')
    return history

if __name__ == '__main__':
    main()
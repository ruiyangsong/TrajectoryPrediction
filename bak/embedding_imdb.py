import json
from keras.datasets import imdb
from keras import preprocessing, layers
from keras.models import Sequential



def main():
    max_features = 10000
    maxlen=20
    x_train, y_train, x_test, y_test = load_imdb(max_features=max_features, maxlen=maxlen)
    print('max_featuews: %s'
          '\nmaxlen_per_review: %s'
          '\nx_train type: %s'
          '\ny_train type: %s'
          '\nx_train shape: %s'
          '\ny_train shape: %s'
          '\nx_test shape: %s'
          '\ny_test shape: %s' % (
              max_features, maxlen, type(x_train), type(y_train), x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    mdoel(x_train,y_train)

def mdoel(x_train, y_train, max_features=10000, maxlen=20, embededlen=100):
    model = Sequential()
    model.add(layers.Embedding(input_dim=max_features, #Size of the vocabulary
                               output_dim=embededlen, #Dimension of the dense embedding.
                               input_length=maxlen, #Length of input sequences
                               ))# 激活的输出shape: (batch, maxlen, enbededlen)
    model.add(layers.Flatten()) #output shape: (samples, maxlen * enbededlen)

    model.add(layers.Dense(1, activation='sigmoid'))#Dense层将每一个单词单独处理，不会考虑单词间的关系，添加循环层和卷积层可以解决此缺点

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)
    return history.history



def decode_reviews(x):
    with open('E:/study/机器学习/dataset/imdb/imdb_word_index.json') as f:
        word_index = json.load(f) # word_index 是将单词映射为整数索引的dict
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #将整数索引映射为单词
    decoded_review = [' '.join([reverse_word_index.get(j - 3, '?') for j in xi]) for xi in x]#索引减去了3，因为0、1、2是为“padding”（填充）、“start of sequence”（序列开始）、“unknown”（未知词）分别保留的索引
    return decoded_review

def load_imdb(path = 'E:/study/机器学习/dataset/imdb/imdb.npz', max_features = 10000, maxlen = 20):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path=path, num_words=max_features) #加载数据集为整数"列表"
    ## 将整数列表转换成形状为(samples, maxlen) 的二维整数张量, 也即(samples, word_indices)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, truncating='pre')  # 默认从后面截断,不够会进行填充
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, truncating='pre')

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    main()
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from src.bak.embedding_imdb import load_imdb
def main():
    max_features = 10000
    maxlen=500
    x_train, y_train, x_test, y_test = load_imdb(max_features=max_features, maxlen=maxlen)
    model(x_train, y_train)

def model(x_train, y_train):
    model = Sequential()
    model.add(Embedding(input_dim=10000,
                        output_dim=100))
    model.add(LSTM(units=64)) # 输入形状为(None,None,100)
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=1,
                        batch_size=128,
                        validation_split=0.2)
    return history

if __name__ == '__main__':
    main()
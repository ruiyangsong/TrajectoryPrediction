'''
对每一个序列来说，RNN遍历序列中的每一个元素（时间步）并保留一个状态，每个时间步的输出是网络当前状态和当前输入的组合，
当前输入为上一时间步的输出，对于第一个时间步而言，其状态需要初始化（如全部初始化为0）

RNN伪代码
state_t = 0 #网络初始状态
for input_t in input_sequence:
    output_t = f(input_t, state_t) # that is output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t

simpleRNN过于简化，没有实用价值。
在时刻t，理论上其能记住许多时间步之前见过的信息，但实际上是不可能学习到这种长期依赖的，其原因是梯度消失问题。
而 LSTM 层和 GRU 层都是为了解决这个问题而设计的。
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.datasets import imdb
from keras.preprocessing import sequence
from src.bak.embedding_imdb import load_imdb

def main():
    max_features = 10000
    maxlen = 20
    x_train, y_train, x_test, y_test = load_imdb(max_features=max_features, maxlen=maxlen)
    history = simpleRNN(x_train, y_train)


def rnn_forward_numpy():
    '''
    rnn 前向传播的numpy实现
    注意这个函数一次只能处理一个序列 并不是批量的
    '''
    timesteps       = 100 #输入序列的时间步数
    input_features  = 32  #输入特征空间的维度
    output_features = 64  #输出特征空间的维度

    inputs = np.random.random((timesteps,input_features))#噪声数据作为输入
    state_t = np.zeros((output_features,))#初始状态为全零向量

    W = np.random.random((output_features, input_features))
    U = np.random.random((output_features, output_features))
    b = np.random.random((output_features,))

    successive_outputs = []
    for input_t in inputs:
        print()
        output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
        successive_outputs.append(output_t) #将输出保存
        state_t = output_t
    final_output_sequence = np.stack(successive_outputs, axis=0) #(timesteps, output_features)

def simpleRNN(x_train, y_train):
    '''
    这个简单的rnn不适合处理长序列
    :return:
    '''
    model = Sequential()
    model.add(Embedding(input_dim=10000,
                        output_dim=32
                        ))# 注意这里没有指定序列长度，但每个批次的序列应当具有相同的长度

    model.add(SimpleRNN(units=64, # units: dimensionality of the output space.
                        return_sequences=True #是否返回每个时间步的输出 return (batch_size, timesteps, output_features) if True, else (batch_size, output_features)
                                              #如果要堆叠多个循环层，则需要中间层都返回完整的输出序列
                        ))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(SimpleRNN(32)) #最后一层返回最终输出(batch_size, output_features)
    # model.add(Flatten()) #本身就是2D张量（每个样本是1D张量），不需要Flatten
    model.add(Dense(1,activation='sigmoid'))
    model.summary()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=1,
                        batch_size=128,
                        validation_split=0.2)

    return history


if __name__ == '__main__':
    main()
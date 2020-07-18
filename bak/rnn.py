import sys

import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from utils import  config_tf, net_saver, net_predictor

global USEGPU
USEGPU = False

def main():
    lookback   = 600  # 考虑过去的600个点中的100个（600/6=100）
    step       = 6  # 采样周期为6秒
    delay      = 6  # 预测6秒后的数据
    batch_size = 128
    train_num  = 20000
    val_num    = 5000
    test_num   = None
    modeldir   = '../model'
    #
    # prepare data
    #
    float_data = parse_data()
    train_gen, _ = gen_data(float_data, lookback, step, delay, batch_size, min_idx=0, max_idx=train_num+1)
    val_gen, val_steps = gen_data(float_data, lookback, step, delay, batch_size, min_idx=train_num+1, max_idx=train_num+val_num+1)
    test_gen, _ = gen_data(float_data, lookback, step, delay, batch_size=5000, min_idx=train_num+val_num+1, max_idx=len(float_data)-delay-1)
    x_test, y_test = next(test_gen)

    #
    # train and save
    #
    if USEGPU:
        config_tf(user_mem=2500, cuda_rate=0.2)
    model, history_dict = GRU(train_gen, val_gen, val_steps, input_dim=4)
    net_saver(model, modeldir, history_dict)

    #
    # test
    #
    y_real, y_pred = net_predictor(modeldir, x_test, y_test, Onsave=True)

    print('\n y_real: %s'
          '\n y_pred: %s'%(y_real, y_pred))

def parse_data():
    '''
    时间（秒），纬度（度），经度（度），高度（米），速度（米/秒），航向（度）
    :return: float arr
    '''
    data_pth = r'../data/data.csv'
    with open(data_pth, encoding='utf-8') as f:
        lines = f.readlines()
    float_data = np.zeros((len(lines), len(lines[0].split(','))-2))
    for i, line in enumerate(lines):
        float_data[i,:] = [float(x) for x in [line.split(',')[i] for i in [1,2,4,5]]]
    print('data shape:', float_data.shape)
    # # plot
    # from matplotlib import pyplot as plt
    # lati = float_data[:, 1]  # 温度（单位：摄氏度）
    # plt.plot(range(len(lati)), lati)
    # plt.show()
    return float_data

def normalization(float_data, train_num=20000):
    '''只基于train来标准化'''
    data = float_data.copy()
    mean = float_data[:train_num].mean(axis=0)
    data -= mean
    std = float_data[:train_num].std(axis=0)
    data /= std

    return data

def gen_data(float_data, lookback, step, delay, batch_size, min_idx, max_idx):
    data_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=min_idx,
                         max_index=max_idx,
                         shuffle=True,
                         step=step,
                         batch_size=batch_size)

    steps = (max_idx-min_idx-1 - lookback) // batch_size
    return data_gen, steps

def generator(float_data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6, train_num=20000):
    '''
    :param float_data: un-normalized data
    :param lookback: 输入数据应该包括过去多少个时间步
    :param delay: 目标应该在未来多少个时间步"之后"
    :param min_index: data 数组中的索引，用于界定需要抽取哪些时间步
    :param max_index: data 数组中的索引，用于界定需要抽取哪些时间步
    :param shuffle: 是打乱样本，还是按顺序抽取样本
    :param batch_size: 每个批量的样本数量
    :param step: 数据采样的周期
    :return:
    '''
    data = normalization(float_data, train_num=train_num)
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            # if i + batch_size >= max_index:
            if i >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), 2))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = float_data[rows[j] + delay][:2]
        # yield rows
        yield samples, targets

def dense(float_data, train_gen, val_gen, val_steps, lookback, step):
    metrics = ('mae',)

    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1]))) # (seq_len, channels)
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=list(metrics))
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    return history

def GRU(train_gen, val_gen, val_steps, input_dim=4):
    metrics = ('mae',)
    model = Sequential()
    model.add(layers.GRU(32, return_sequences=True, input_shape=(None, input_dim)))#3D tensor with shape `(batch_size, timesteps, input_dim)`.
    model.add(layers.LSTM(32, return_sequences=True,))#3D tensor with shape `(batch_size, timesteps, input_dim)`.
    model.add(layers.LSTM(64, return_sequences=False,))#3D tensor with shape `(batch_size, timesteps, input_dim)`.
    model.add(layers.Dense(2))
    model.summary()
    model.compile(optimizer=RMSprop(), loss='mse', metrics=list(metrics))
    result = model.fit_generator(train_gen,
                                 steps_per_epoch=200,#每一个epoch抽取几次gen数据, It should typically be equal to the number of samples of your dataset divided by the batch size.
                                 epochs=50,
                                 validation_data=val_gen,
                                 validation_steps=val_steps)
    return model, result.history

if __name__ == '__main__':
    main()

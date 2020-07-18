import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

def main():
    float_data = parse_data()
    float_data = normalization(float_data)
    train_gen, test_gen, val_gen, val_steps, test_steps = gen_data(float_data)
    GRU(float_data, train_gen, val_gen, val_steps)

def parse_data():
    data_pth = r'E:/study/机器学习/dataset/climate/jena_climate_2009_2016.csv'
    with open(data_pth, encoding='utf-8') as f:
        lines = f.readlines()[1:]
    float_data = np.zeros((len(lines), len(lines[0].split(','))-1))
    for i, line in enumerate(lines):
        float_data[i,:] = [float(x) for x in line.split(',')[1:]]

    ## plot
    # from matplotlib import pyplot as plt
    # temp = float_data[:, 1]  # 温度（单位：摄氏度）
    # plt.plot(range(len(temp)), temp)
    # plt.show()

    return float_data

def normalization(float_data, train_num=200000):
    '''只基于train来标准化'''
    mean = float_data[:train_num].mean(axis=0)
    float_data -= mean
    std = float_data[:train_num].std(axis=0)
    float_data /= std

    return float_data

def gen_data(float_data):
    lookback   = 1440 #过去10天的数据（因为数据是每10分钟采一次样）
    step       = 6 #采样周期
    delay      = 144 #预测一天后的数据
    batch_size = 128

    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen   = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=200001,
                          max_index=300000,
                          step=step,
                          batch_size=batch_size)
    test_gen  = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=300001,
                          max_index=None,
                          step=step,
                          batch_size=batch_size)
    val_steps = (300000 - 200001 - lookback) // batch_size
    test_steps = (len(float_data) - 300001 - lookback) // batch_size
    return train_gen, test_gen, val_gen, val_steps, test_steps

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    '''
    :param data: normalized data
    :param lookback: 输入数据应该包括过去多少个时间步
    :param delay: 目标应该在未来多少个时间步"之后"
    :param min_index: data 数组中的索引，用于界定需要抽取哪些时间步
    :param max_index: data 数组中的索引，用于界定需要抽取哪些时间步
    :param shuffle: 是打乱样本，还是按顺序抽取样本
    :param batch_size: 每个批量的样本数量
    :param step: 数据采样的周期，每一个小时抽取一个样本点
    :return:
    '''
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    cnt=0
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
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        # yield rows
        yield samples, targets
        if max_index <= 200001:
            cnt+=1
            print('\ntrain_gen were called: %s times'%cnt)

def dense(float_data, train_gen, val_gen, val_steps, lookback=1440, step=6):
    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1]))) # (seq_len, channels)
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    return history

def GRU(float_data, train_gen, val_gen, val_steps):
    model = Sequential()
    model.add(layers.GRU(32, input_shape=(240, float_data.shape[-1])))#3D tensor with shape `(batch_size, timesteps, input_dim)`.
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=10,#每一个epoch抽取几次gen数据, It should typically be equal to the number of samples of your dataset divided by the batch size.
                                  epochs=1,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    return history

if __name__ == '__main__':
    main()

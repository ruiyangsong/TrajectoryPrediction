'''
Train rnn model from given trajectories, for help information --> $ python rnn.py -h
1. train data are located at '../data/train'
2. mean and std of train data are saved at '../data/train'
3. trained model is saved at '../model'
@Ruiyang Song
'''
import os, argparse
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from utils import config_tf, net_saver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path',   type=str, required=True,     help='[str], list file path for train trajectories, each row in the list file represents a trajectory')
    parser.add_argument('--valid_ratio', type=float, default=0.2,     help='[float], split @valid_ratio data from train_data as validation set, default value is "0.2"')
    parser.add_argument('--lookback',    type=int,   default=600,     help='[int], how long the historical data be seen back, default value is "600 (seconds)"')
    parser.add_argument('--step',        type=int,   default=6,       help='[int], sampling interval, default value is "6 (seconds)"')
    parser.add_argument('--delay',       type=int,   default=6,       help='[int], predicting data after @delay seconds, default value is "6 (seconds)"')
    parser.add_argument('--model_name',  type=str,   default='model', help='[str], the mame of your trained model, default value is "model"')
    parser.add_argument('--batch_size',  type=int,   default=128,     help='[int], training batch_size, dafault value is "128"')
    parser.add_argument('--epochs',      type=int,   default=50,      help='[int], training epochs, dafault value is "50"')
    parser.add_argument('--gpu_mem',     type=int,   default=0,       help='[int], whether GPU memory is used or not, dafault value is "0 (Mib)"')
    args = parser.parse_args()

    list_path   = args.list_path
    valid_ratio = args.valid_ratio
    lookback    = args.lookback # 考虑过去的 600 个点中的100个（600/6=100）[lookback=600, step=6]
    step        = args.step # 采样周期（秒）
    delay       = args.delay # 预测 6 秒后的数据[delay=6]
    model_name  = args.model_name
    batch_size  = args.batch_size
    epochs      = args.epochs
    gpu_mem     = args.gpu_mem

    modeldir    = '../model'
    os.makedirs(modeldir, exist_ok=True)

    #
    # prepare data
    #
    trajectory_dict = parse_data(list_path)
    normalized_trajectory_dict = normalization(trajectory_dict, valid_ratio=valid_ratio)

    #
    # train and save
    #
    if gpu_mem > 1:
        config_tf(user_mem=gpu_mem)

    for k, v in normalized_trajectory_dict.items():
        train_num = round(v.shape[0] * (1-valid_ratio))
        train_gen = generator(data=v, lookback=lookback, delay=delay, step=step, min_index=0, max_index=train_num+1,
                              shuffle=False, batch_size=batch_size)
        steps_per_epoch = (train_num - lookback) // batch_size

        valid_gen = generator(data=v, lookback=lookback, delay=delay, step=step, min_index=train_num+1, max_index=v.shape[0]-delay-1,
                              shuffle=False, batch_size=batch_size)
        val_steps = (len(v) - delay - 1 - train_num -1 - 1 - lookback) // batch_size

        model, history_dict = RNN(train_gen, valid_gen, val_steps, input_dim=v.shape[1], epochs=epochs,
                                  steps_per_epoch=steps_per_epoch)

    net_saver(model, modeldir, model_name, history_dict)


def parse_data(list_path):
    '''
    列名称 --> [时间（秒），纬度（度），经度（度），高度（米），速度（米/秒），航向（度）]
    :param list_path: where file that trajectories listed in
    :return: trajectory_dict
    '''
    trajectory_dict = {}
    with open(list_path,'r') as f:
        trajectories_lst = [x.strip() for x in f.readlines()]
    for trajectory in trajectories_lst:
        data_pth = '../data/train/%s.csv'%trajectory
        with open(data_pth, encoding='utf-8') as f:
            lines = f.readlines()
        float_data = np.zeros((len(lines), len(lines[0].split(','))-2))
        for i, line in enumerate(lines):
            float_data[i,:] = [float(x) for x in [line.split(',')[i] for i in [1,2,4,5]]]
        trajectory_dict[trajectory] = float_data
        print('shape of %s: %s' %(trajectory, float_data.shape))

    return trajectory_dict


def normalization(trajectory_dict, valid_ratio):
    '''基于train来标准化数据'''
    train_data_all = np.asarray([v[:round(v.shape[0] * (1 - valid_ratio))] for v in trajectory_dict.values()]).reshape(-1,list(trajectory_dict.values())[0].shape[1])
    mean = train_data_all.mean(axis=0)
    std  = train_data_all.std(axis=0)
    normalized_trajectory_dict = {}
    for k, v in trajectory_dict.items():
        v -= mean
        v /= std
        normalized_trajectory_dict[k] = v
    with open('../data/train/mean_std.txt','w') as f:
        f.write('mean: %s\n'%list(mean))
        f.write('std: %s\n'%list(std))

    return normalized_trajectory_dict


def generator(data, lookback, delay, step=6, min_index=0, max_index=None, shuffle=False, batch_size=128):
    '''
    :param data: data normalized by train mean and std
    :param lookback: 输入数据应该包括过去多少个时间步
    :param delay: 目标应该在未来多少个时间步"之后"
    :param step: 数据采样的周期
    :param min_index: data 数组中的索引，用于界定需要抽取哪些时间步
    :param max_index: data 数组中的索引，用于界定需要抽取哪些时间步
    :param shuffle: 是打乱样本，还是按顺序抽取样本
    :param batch_size: 每个批量的样本数量
    :return:
    '''
    if max_index is None:
        max_index = len(data) - delay - 1
    assert max_index > (lookback + delay + 1)
    flag = 0
    i = min_index + lookback + flag
    while 1:
        if shuffle:
            rows = sorted(np.random.randint(min_index + lookback, max_index, size=batch_size))
        else:
            # if i + batch_size >= max_index:
            if i >= max_index:
                flag += 1
                i = min_index + lookback + flag
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))#3D array sample
        targets = np.zeros((len(rows), 2))#2D array labels
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][:2]

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


def RNN(train_gen, val_gen, val_steps, input_dim=4, epochs=50, steps_per_epoch=200):
    '''
    :param train_gen:
    :param val_gen:
    :param val_steps:
    :param input_dim:
    :param epochs:
    :param steps_per_epoch: 每一个epoch抽取几次gen数据,
                            It should typically be equal to the number of samples of your dataset divided by the batch size.
    :return:
    '''
    metrics = ('mae',)
    model = Sequential()
    model.add(layers.GRU(32, return_sequences=True, input_shape=(None, input_dim)))#3D tensor with shape `(batch_size, timesteps, input_dim)`.
    model.add(layers.LSTM(32, return_sequences=True,))#3D tensor with shape `(batch_size, timesteps, input_dim)`.
    model.add(layers.LSTM(64, return_sequences=False,))#3D tensor with shape `(batch_size, timesteps, input_dim)`.
    model.add(layers.Dense(2))
    model.summary()
    model.compile(optimizer=RMSprop(), loss='mse', metrics=list(metrics))
    result = model.fit_generator(train_gen,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 validation_data=val_gen,
                                 validation_steps=val_steps)
    return model, result.history


if __name__ == '__main__':
    main()

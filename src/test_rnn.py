'''
Test rnn model from given trajectories, for help information --> $ python rnn.py -h
1. test data are located at '../data/test'
2. predicted results are saved at '../data/test'
3. mean and std of train data are saved at '../data/train'
4. trained model is saved at '../model'
@Ruiyang Song
'''
import argparse
import numpy as np
from utils import load_model, net_predictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path',   type=str, required=True,     help='[str], list file path for test trajectories, each row represents a trajectory')
    parser.add_argument('--lookback',    type=int,   default=600,     help='[int], how long the historical data be seen back, default value is "600 (seconds)"')
    parser.add_argument('--step',        type=int,   default=6,       help='[int], sampling interval, default value is "6 (seconds)"')
    parser.add_argument('--delay',       type=int,   default=6,       help='[int], predicting data after @delay seconds, default value is "6 (seconds)"')
    parser.add_argument('--model_name',  type=str,   default='model', help='[str], the mame of your trained model, default value is "model"')
    args = parser.parse_args()

    list_path   = args.list_path
    lookback    = args.lookback # 考虑过去的 600 个点中的100个（600/6=100）[lookback=600, step=6]
    step        = args.step # 采样周期（秒）
    delay       = args.delay # 预测 6 秒后的数据[delay=6]
    model_name  = args.model_name

    modeldir    = '../model'

    #
    # prepare data
    #
    trajectory_dict = parse_data(list_path)
    normalized_trajectory_dict = normalization(trajectory_dict)

    #
    # test
    #
    model = load_model(modeldir, model_name)
    for k, v in normalized_trajectory_dict.items():
        test_gen = generator(data=v, lookback=lookback, delay=delay, step=step, min_index=0, max_index=None,
                             shuffle=False, batch_size=10)
        x_test, y_test = next(test_gen)
        print('\nx_test shape:', x_test.shape)
        outpth = '../data/test/%s-rst.npz'%k
        y_pred = net_predictor(model, x_test, y_test, outpth, Onsave=True)
        print('\n y_real: %s'
              '\n y_pred: %s' % (y_test, y_pred))


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
        data_pth = '../data/test/%s.csv'%trajectory
        with open(data_pth, encoding='utf-8') as f:
            lines = f.readlines()
        float_data = np.zeros((len(lines), len(lines[0].split(','))-2))
        for i, line in enumerate(lines):
            float_data[i,:] = [float(x) for x in [line.split(',')[i] for i in [1,2,4,5]]]
        trajectory_dict[trajectory] = float_data
        print('\nshape of %s: %s' %(trajectory, float_data.shape))

    return trajectory_dict


def normalization(trajectory_dict):
    '''基于train来标准化数据'''
    with open('../data/train/mean_std.txt','r') as f:
          lines = [x.strip() for x in f.readlines()]
    mean = eval(lines[0].split(':')[-1].strip())
    std  = eval(lines[1].split(':')[-1].strip())
    normalized_trajectory_dict = {}
    for k, v in trajectory_dict.items():
        v -= mean
        v /= std
        normalized_trajectory_dict[k] = v

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
    flag = 0
    assert max_index > (lookback+delay+1)
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


if __name__ == '__main__':
    main()


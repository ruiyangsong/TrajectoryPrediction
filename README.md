# 此文档用于解释航迹预测程序的执行流程
<center>RSong</center>

此程序基于循环神经网络（GRU和LSTM）从历史航迹数据中学习运动模式并用于预测未知的航迹，
程序的输入为固定长度的历史轨迹点序列（由初始航迹通过滑窗得到），
经过一层GRU和两层LSTM以及一层全连接层后得到预测输出，
输出结果为下一时间间隔（考虑的为6秒）的目标位置（由经纬度坐标描述），
除网络结构外，其中重要的超参数为“序列长度”，其决定了模式的挖掘范围，
不同的序列长度可以从历史数据中构造不同的训练样本。

## 依赖项
```text
|library                   version|
|---------------------------------|
|python                    3.7.3  |
|numpy                     1.16.4 |
|matplotlib                3.1.3  |
|tensorflow                1.13.1 |
|keras                     2.2.4  |
```

## 目录树
```text
TrajectoryPrediction:.
│  README.md
│  
├─data
│  │  generate_data.m
│  │  空中水面水下航迹列标题.txt
│  │  
│  ├─test
│  │      test.lst
│  │      trajectory01-rst.npz
│  │      trajectory01.csv
│  │      
│  └─train
│          mean_std.txt
│          train.lst
│          trajectory01.csv
│          
├─log
│      rnn-20200618.log
│      
├─model
│      model-history.dict
│      model-lossFigure.png
│      model-weightsFinal.h5
│      model.json
│      model.png
│      
└─src
    │  rnn.py
    │  test_rnn.py
    │  utils.py
```

## Train
### 查看帮助信息
```shell script
$ python rnn.py --help

usage: rnn.py [-h] --list_path LIST_PATH [--valid_ratio VALID_RATIO]
              [--lookback LOOKBACK] [--step STEP] [--delay DELAY]
              [--model_name MODEL_NAME] [--batch_size BATCH_SIZE]
              [--epochs EPOCHS] [--gpu_mem GPU_MEM]

optional arguments:
  -h, --help            show this help message and exit
  --list_path LIST_PATH
                        [str], list file path for train trajectories, each row
                        in the list file represents a trajectory
  --valid_ratio VALID_RATIO
                        [float], split @valid_ratio data from train_data as
                        validation set, default value is "0.2"
  --lookback LOOKBACK   [int], how long the historical data be seen back,
                        default value is "600 (seconds)"
  --step STEP           [int], sampling interval, default value is "6
                        (seconds)"
  --delay DELAY         [int], predicting data after @delay seconds, default
                        value is "6 (seconds)"
  --model_name MODEL_NAME
                        [str], the mame of your trained model, default value
                        is "model"
  --batch_size BATCH_SIZE
                        [int], training batch_size, dafault value is "128"
  --epochs EPOCHS       [int], training epochs, dafault value is "50"
  --gpu_mem GPU_MEM     [int], whether GPU memory is used or not, dafault
                        value is "0 (Mib)"
```
### example
```shell script
$ cd src/
$ python rnn.py --list_path ../data/train/train.lst --epochs 50 > ../log/you_log_name.log
```
## Test
### 查看帮助信息
```shell script
$ python test_rnn.py --help

usage: test_rnn.py [-h] --list_path LIST_PATH [--lookback LOOKBACK]
                   [--step STEP] [--delay DELAY] [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --list_path LIST_PATH
                        [str], list file path for test trajectories, each row
                        represents a trajectory
  --lookback LOOKBACK   [int], how long the historical data be seen back,
                        default value is "600 (seconds)"
  --step STEP           [int], sampling interval, default value is "6
                        (seconds)"
  --delay DELAY         [int], predicting data after @delay seconds, default
                        value is "6 (seconds)"
  --model_name MODEL_NAME
                        [str], the mame of your trained model, default value
                        is "model"
```
### example
```shell script
$ cd src/
$ python test_rnn.py --list_path ../data/test/test.lst
```
## 网络
### 参数量
```text
data shape: (33321, 4)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
gru_1 (GRU)                  (None, None, 32)          3552      
_________________________________________________________________
lstm_1 (LSTM)                (None, None, 32)          8320      
_________________________________________________________________
lstm_2 (LSTM)                (None, 64)                24832     
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 130       
=================================================================
Total params: 36,834
Trainable params: 36,834
Non-trainable params: 0
```

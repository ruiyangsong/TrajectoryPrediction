import keras as ks
import numpy as np
'''
对于变长输入
方法一
1. 首先应保证模型对不同长度的输入具有相同数量的参数
2. 保证每个epochde的数据具有相同的长度
   可以通过多次fit或者使用fit_gen实现

方法二（对于可以使用Masking的后续层）
1. 初始输入包含0
2. 利用Masking层mask掉0
    For each timestep in the input tensor (dimension #1 in the tensor),
    if all values in the input tensor at that timestep
    are equal to `mask_value`, then the timestep will be masked (skipped)
    in all downstream layers (as long as they support masking).

    If "any" downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.
'''
def method_1():
    ## 长度均为3
    input_array = np.array([[4, 10, 5], [2, 1, 6], [3, 7, 9], [2, 5, 3]])
    input_array = input_array[:,:,np.newaxis]
    ## 长度均为2
    input_array_2 = np.array([[10, 5], [1, 6], [7, 9], [5, 3]])
    input_array_2 = input_array_2[:,:,np.newaxis]

    print(input_array.shape)
    print(input_array_2.shape)

    y_arr = np.array([1,0,1,0])

    model = ks.models.Sequential()
    model.add(ks.layers.Conv1D(filters=32,kernel_size=2,padding='same', input_shape=(None, 1)))#不指定序列长度
    model.add(ks.layers.GRU(units=1, return_sequences=False, input_shape=(None,32)))#不指定序列长度
    model.add(ks.layers.Dense(1))
    model.summary()

    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
    model.fit(x=input_array,
              y=y_arr,
              epochs=1,
              batch_size=1,
              )
    model.fit(x=input_array_2,
              y=y_arr,
              epochs=1,
              batch_size=1,
              )

    ## 同一个模型预测两种长度的数据
    output_array = model.predict(input_array)
    output_array_2 = model.predict(input_array_2)
    print('==========================================')
    print(output_array)
    print('==========================================')
    print(output_array_2)

def method_2():
    input_array = np.array([[4, 10, 5], [2, 1, 0], [3, 7, 9], [2, 5, 3]])
    input_array = input_array[:,:,np.newaxis]
    ## 长度均为2
    input_array_2 = np.array([[10, 5], [1, 6], [7, 9], [5, 3]])
    input_array_2 = input_array_2[:, :, np.newaxis]

    print(input_array.shape)
    print(input_array_2.shape)

    y_arr = np.array([1, 0, 1, 0])

    model = ks.models.Sequential()
    model.add(ks.layers.Masking(mask_value=0, input_shape=(None, 1)))
    # model.add(ks.layers.Conv1D(filters=32, kernel_size=2, padding='same', input_shape=(None, 1)))  # Conv1D不支持masking
    model.add(ks.layers.GRU(name='gru',units=1, return_sequences=False, input_shape=(None, 1)))  # 不指定序列长度,支持masking
    model.add(ks.layers.Dense(1))
    model.summary()

    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])

    model.fit(x=input_array,
              y=y_arr,
              epochs=1,
              batch_size=1,
              )
    model.fit(x=input_array_2,
              y=y_arr,
              epochs=1,
              batch_size=1,
              )
    ## 同一个模型预测两种长度的数据
    output_array = model.predict(input_array)
    output_array_2 = model.predict(input_array_2)
    print('==========================================')
    print(output_array)
    print('==========================================')
    print(output_array_2)

if __name__ == '__main__':
    # method_1() #runs good
    method_2() #runs good
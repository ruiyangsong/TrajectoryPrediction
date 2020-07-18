import os, time
import numpy as np

def loss_plot(history_dict, outpth):
    from matplotlib import pyplot as plt
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']
    mae      = history_dict['mean_absolute_error']
    val_mae  = history_dict['val_mean_absolute_error']
    epochs   = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 5), dpi=100)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mae, 'r', label='Training mae')
    plt.plot(epochs, val_mae, 'g', label='Validation mae')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.legend(loc="lower right")
    # plt.show()

    plt.savefig(outpth)#pngfile
    plt.clf()


def net_saver(model, modeldir, model_name, history_dict):
    os.makedirs(modeldir, exist_ok=True)
    ## save model architecture
    try:
        model_json = model.to_json()
        with open('%s/%s.json' % (modeldir,model_name), 'w') as json_file:
            json_file.write(model_json)
    except:
        print('save model.json to json failed.')

    ## save model weights
    try:
        model.save_weights(filepath='%s/%s-weightsFinal.h5' % (modeldir, model_name))
    except:
        print('save final model weights failed.')

    ## save training history
    try:
        with open('%s/%s-history.dict' % (modeldir,model_name), 'w') as file:
            file.write(str(history_dict))
        # with open('%s/fold_%s_history.dict'%(modeldir,k_count), 'r') as file:
        #     print(eval(file.read()))
    except:
        print('save history_dict failed.')

    ## save loss figure
    try:
        figure_pth = '%s/%s-lossFigure.png' %(modeldir, model_name)
        loss_plot(history_dict, outpth=figure_pth)
    except:
        print('save loss plot figure failed.')

    # save model figure
    try:
        from keras.utils import plot_model
        plot_model(model, to_file='%s/%s.png'%(modeldir,model_name), show_shapes=True, show_layer_names=True, dpi=200)
    except:
        print('save model figure failed.')

def load_model(modeldir, model_name):
    from keras import models
    # Load model
    with open('%s/%s.json' % (modeldir, model_name), 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = models.model_from_json(loaded_model_json)  # keras.models.model_from_yaml(yaml_string)
    loaded_model.load_weights(filepath='%s/%s-weightsFinal.h5' % (modeldir,model_name))
    return loaded_model

def net_predictor(model, x_test, y_test, outpth=None, Onsave=True):
    y_pred = model.predict(x_test, batch_size=32, verbose=0)  # prob ndarray
    #
    # save x_test and y_real and y_pred
    #
    if Onsave:
        try:
           np.savez(outpth, x_test=x_test, y_real=y_test, y_pred=y_pred)
        except:
            print('save test_rst failed')

    return y_pred


def config_tf(user_mem=2500, cuda_rate=0.01):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    queueGPU(USER_MEM=user_mem, INTERVAL=60)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if cuda_rate != 'full':
        config = tf.ConfigProto()
        if float(cuda_rate) < 0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(cuda_rate)
        set_session(tf.Session(config=config))


def queueGPU(USER_MEM=10000,INTERVAL=60,Verbose=1):
    """
    :param USER_MEM: int, Memory in Mib that your program needs to allocate
    :param INTERVAL: int, Sleep time in second
    :return:
    """
    try:
        totalmemlst=[int(x.split()[2]) for x in os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total').readlines()]
        assert USER_MEM<=max(totalmemlst)
    except:
        print('\033[1;35m[WARNING]\nUSER_MEM should smaller than one of the GPU_TOTAL --> %s MiB.\nReset USER_MEM to %s MiB.\033[0m'%(totalmemlst,max(totalmemlst)-1))
        USER_MEM=max(totalmemlst)-1
    while True:
        memlst=[int(x.split()[2]) for x in os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()]
        if Verbose:
            os.system("echo 'Check at:' `date`")
            print('GPU Free Memory List --> %s MiB'%memlst)
        idxlst=sorted(range(len(memlst)), key=lambda k: memlst[k])
        boollst=[y>USER_MEM for y in sorted(memlst)]
        try:
            GPU=idxlst[boollst.index(True)]
            os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
            print('GPU %s was chosen.'%GPU)
            break
        except:
            time.sleep(INTERVAL)


if __name__ == '__main__':
    figure_pth = '../model/lossFigure.png'
    hist_pth   = '../model/history.dict'
    with open(hist_pth) as f:
        history_dict = eval(f.read())
    print(history_dict)
    loss_plot(history_dict, outpth=figure_pth)

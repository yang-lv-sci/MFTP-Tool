import math
import os
import numpy as np
#from tensorflow import set_random_seed
import tensorflow as tf
from test1 import test_main
from tensorflow.compat.v1.keras.backend import set_session
import time
from pathlib import Path
from model import model_base

peptide_type = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']
tf.random.set_seed(101)
np.random.seed(101)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))


def counters(y_train):
    # counting the number of each peptide
    counterx = np.zeros(len(peptide_type) + 1, dtype='int')
    for i in y_train:
        a = np.sum(i)
        a = int(a)
        counterx[a] += 1
    print(counterx)

def train_method(train,test, para, model_num, model_path, data_size):
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=100,mode="max")

    reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10,

                           verbose=1,

                           factor=0.8,

                           min_lr=1e-6)

    # Implementation of training method
    Path(model_path).mkdir(exist_ok=True)

    # data get
    X_train, y_train = train[0], train[1]
    
    X_test, y_test = test[0], test[1]

    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    counters(y_train)

    # train
    length = X_train.shape[1]
    width= X_train.shape[2]
    out_length = y_train.shape[1]

    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon, t_data.tm_mday, t_data.tm_hour,
                                                                      t_data.tm_min, t_data.tm_sec))

    class_weights = {}

    sumx = len(X_train)

    # 定义回调

    for m in range(len(data_size)):
        g = 5 * math.pow(int((math.log((sumx / data_size[m]), 2))), 2)
        if g <= 0:
            g = 1
        x = {m: g}
        class_weights.update(x)

    for counter in range(8, 7+model_num + 1):
        
        each_model = os.path.join(model_path, 'model' + "_" + str(counter) + '.h5')
        
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=each_model,

                                   monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        #model = model_base(length, out_length, para)
        model = model_base(length,width, out_length, para)

        model.fit(X_train, y_train, epochs=10000, batch_size=64, verbose=1,
                  validation_data=(X_test, y_test),
                  shuffle=True,callbacks=[checkpointer,earlystop_callback,reduce],class_weight=class_weights)  # class_weight=class_weights

        """each_model = os.path.join(model_path, 'model' + "_" + str(counter) + '.h5')

        model.save(each_model)"""

        tt = time.localtime(time.time())
        with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
            f.write('count{}: {}m {}d {}h {}m {}s\n'.format(str(counter), tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min,
                                                            tt.tm_sec))


def train_main(train, test, model_num, modelDir, data_size):
    # parameters
    ed = 128
    ps = 5
    fd = 128
    dp = 0.6
    lr = 8e-4

    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    # Conduct training
    train_method(train,test, para, model_num, modelDir, data_size)

    # prediction
    test_main(test, para, model_num, modelDir)

    tt = time.localtime(time.time())
    with open(os.path.join(modelDir, 'time.txt'), 'a+') as f:
        f.write(
            'test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
from keras import initializers
from keras import backend as K
from tensorflow.python.keras.layers import Layer
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout,Reshape
from keras.layers import Flatten, Dense, CuDNNLSTM
from tensorflow.keras import Model
from keras.regularizers import l2,l1
from keras.optimizers import adam_v2
from keras.layers.wrappers import Bidirectional
import numpy as np


class MultiHeadAttention(Layer):
    def __init__(self, output_dim, num_head, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        return {"output_dim": self.output_dim, "num_head": self.num_head}

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.built = True

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])
        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
        e = e / (self.output_dim ** 0.5)
        e = K.softmax(e)
        outputs = K.batch_dot(e, v)
        for i in range(1, self.W.shape[0]):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            # print('q_shape:'+str(q.shape))
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            # print('e_shape:'+str(e.shape))
            o = K.batch_dot(e, v)
            outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def model_base(length,width, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,width), dtype='float32', name='main_input')
    first_input=main_input[:,:,1:]
    second_input=main_input[:,:,0]
    
    """LY"""
    a = Convolution1D(128, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(first_input)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)

    b = Convolution1D(128, 4, activation='relu', padding='same', kernel_regularizer=l2(l2value))(first_input)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)

    c = Convolution1D(128, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(first_input)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)
    
    """Paper"""
    second_input = Embedding(output_dim=ed, input_dim=21, input_length=length, name='Embadding')(second_input)
    
    d = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(second_input)
    dpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(d)

    e = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(second_input)
    epool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(e)

    f = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(second_input)
    fpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(f)
    
    merge = Concatenate(axis=-1)([apool,bpool,cpool,dpool,epool,fpool])
    #LYmerge = Concatenate(axis=-1)([apool,bpool,cpool])
    
    merge = Dropout(dp)(merge)
    #LYmerge = Dropout(dp)(LYmerge)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(merge)

    x = MultiHeadAttention(80, 5)(x)
    
    x = Flatten()(x)
    #LY=Flatten()(LYmerge)
    #x=Concatenate(axis=-1)([LY,x])
    
    x = Dense(fd, activation='relu', kernel_regularizer=l2(l2value))(x)
    

    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, outputs=output)
    adam = adam_v2.Adam(lr=lr,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

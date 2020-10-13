"""
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#This part realizes the quantization and dequantization operations.
#The output of the encoder must be the bitstream.


def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, 4:]).reshape(-1,
                                                                                                            Num_.shape[
                                                                                                                1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config

@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config



def Encoder(x,feedback_bits):
    B=4
    with tf.compat.v1.variable_scope('Encoder'):
        # feature map is (14, 30,)
        stem_conv = layers.Conv2D(32, 3, activation="relu")(x)
        stem_bn = layers.BatchNormalization()(stem_conv)
        block1a_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(stem_bn)
        block1a_bn = layers.BatchNormalization()(block1a_dwconv)
        block1a_se_squeeze = layers.GlobalAveragePooling2D()(block1a_bn)
        block1a_se_reshape = layers.Reshape((1, 1, 32))(block1a_se_squeeze)
        block1a_se_reduce = layers.Conv2D(8, 1, padding = 'SAME', activation="relu")(block1a_se_reshape)
        block1a_se_expand = layers.Conv2D(32, 1, padding = 'SAME', activation="relu")(block1a_se_reduce)
        block1a_se_excite = layers.Multiply()([block1a_bn, block1a_se_expand])

        block1a_project_conv = layers.Conv2D(16, 3, padding = 'SAME',activation="relu")(block1a_se_excite)
        block1a_project_bn = layers.BatchNormalization()(block1a_project_conv)
        block2a_expand_conv = layers.Conv2D(96, 1, padding = 'SAME',activation="relu")(block1a_project_bn)
        block2a_expand_bn = layers.BatchNormalization()(block2a_expand_conv)
        # feature map is (12, 28,)
        block2a_dwconv = layers.DepthwiseConv2D(3, activation="relu")(block2a_expand_bn)
        block2a_bn = layers.BatchNormalization()(block2a_dwconv)
        block2a_se_squeeze = layers.GlobalAveragePooling2D()(block2a_bn)
        block2a_se_reshape = layers.Reshape((1, 1, 96))(block2a_se_squeeze)
        block2a_se_reduce = layers.Conv2D(4, 1, padding = 'SAME', activation="relu")(block2a_se_reshape)
        block2a_se_expand = layers.Conv2D(96, 1, padding = 'SAME', activation="relu")(block2a_se_reduce)
        block2a_se_excite = layers.Multiply()([block2a_bn, block2a_se_expand])

        block2a_project_conv = layers.Conv2D(24, 3, padding = 'SAME',activation="relu")(block2a_se_excite)
        block2a_project_bn = layers.BatchNormalization()(block2a_project_conv)
        block2b_expand_conv = layers.Conv2D(144, 1, padding = 'SAME',activation="relu")(block2a_project_bn)
        block2b_expand_bn = layers.BatchNormalization()(block2b_expand_conv)
        block2b_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block2b_expand_bn)
        block2b_bn = layers.BatchNormalization()(block2b_dwconv)
        block2b_se_squeeze = layers.GlobalAveragePooling2D()(block2b_bn)
        block2b_se_reshape = layers.Reshape((1, 1, 144))(block2b_se_squeeze)
        block2b_se_reduce = layers.Conv2D(6, 1, padding = 'SAME', activation="relu")(block2b_se_reshape)
        block2b_se_expand = layers.Conv2D(144, 1, padding = 'SAME', activation="relu")(block2b_se_reduce)
        block2b_se_excite = layers.Multiply()([block2b_bn, block2b_se_expand])


        block2b_project_conv = layers.Conv2D(24, 3, padding = 'SAME',activation="relu")(block2b_se_excite)
        block2b_project_bn = layers.BatchNormalization()(block2b_project_conv)
        # delte dropout
        block3a_expand_conv = layers.Conv2D(144, 1, padding = 'SAME',activation="relu")(block2b_project_bn)
        block3a_expand_bn = layers.BatchNormalization()(block3a_expand_conv)
        # feature map is (10, 26,)
        block3a_dwconv = layers.DepthwiseConv2D(3, activation="relu")(block3a_expand_bn)
        block3a_bn = layers.BatchNormalization()(block3a_dwconv)
        block3a_se_squeeze = layers.GlobalAveragePooling2D()(block3a_bn)
        block3a_se_reshape = layers.Reshape((1, 1, 144))(block3a_se_squeeze)
        block3a_se_reduce = layers.Conv2D(6, 1, padding = 'SAME', activation="relu")(block3a_se_reshape)
        block3a_se_expand = layers.Conv2D(144, 1, padding = 'SAME', activation="relu")(block3a_se_reduce)
        block3a_se_excite = layers.Multiply()([block3a_bn, block3a_se_expand])


        block3a_project_conv = layers.Conv2D(40, 3, padding = 'SAME',activation="relu")(block3a_se_excite)
        block3a_project_bn = layers.BatchNormalization()(block3a_project_conv)
        block3b_expand_conv = layers.Conv2D(240, 1, padding = 'SAME',activation="relu")(block3a_project_bn)
        block3b_expand_bn = layers.BatchNormalization()(block3b_expand_conv)
        block3b_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block3b_expand_bn)
        block3b_bn = layers.BatchNormalization()(block3b_dwconv)
        block3b_se_squeeze = layers.GlobalAveragePooling2D()(block3b_bn)
        block3b_se_reshape = layers.Reshape((1, 1, 240))(block3b_se_squeeze)
        block3b_se_reduce = layers.Conv2D(10, 1, padding = 'SAME',activation="relu")(block3b_se_reshape)
        block3b_se_expand = layers.Conv2D(240, 1, padding = 'SAME', activation="relu")(block3b_se_reduce)
        block3b_se_excite = layers.Multiply()([block3b_bn, block3b_se_expand])


        block3b_project_conv = layers.Conv2D(40, 3, padding = 'SAME',activation="relu")(block3b_se_excite)
        block3b_project_bn = layers.BatchNormalization()(block3b_project_conv)
        # delte dropout
        block4a_expand_conv = layers.Conv2D(240, 3, padding = 'SAME',activation="relu")(block3b_project_bn)
        block4a_expand_bn = layers.BatchNormalization()(block4a_expand_conv)

        
        block4a_project_conv = layers.Conv2D(80, 3, padding = 'SAME',activation="relu")(block4a_expand_bn)
        block4a_project_bn = layers.BatchNormalization()(block4a_project_conv)
        block4b_expand_conv = layers.Conv2D(480, 1, padding = 'SAME',activation="relu")(block4a_project_bn)
        block4b_expand_bn = layers.BatchNormalization()(block4b_expand_conv)
        block4b_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block4b_expand_bn)
        block4b_bn = layers.BatchNormalization()(block4b_dwconv)
        block4b_se_squeeze = layers.GlobalAveragePooling2D()(block4b_bn)
        block4b_se_reshape = layers.Reshape((1, 1, 480))(block4b_se_squeeze)
        block4b_se_reduce = layers.Conv2D(20, 1, padding = 'SAME',activation="relu")(block4b_se_reshape)
        block4b_se_expand = layers.Conv2D(480, 1, padding = 'SAME', activation="relu")(block4b_se_reduce)
        block4b_se_excite = layers.Multiply()([block4b_bn, block4b_se_expand])


        block4b_project_conv = layers.Conv2D(80, 3, padding = 'SAME',activation="relu")(block4b_se_excite)
        block4b_project_bn = layers.BatchNormalization()(block4b_project_conv)
        # delte dropout
        block4c_expand_conv = layers.Conv2D(480, 1, padding = 'SAME',activation="relu")(block4b_project_bn)
        block4c_expand_bn = layers.BatchNormalization()(block4c_expand_conv)
        block4c_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block4c_expand_bn)
        block4c_bn = layers.BatchNormalization()(block4c_dwconv)
        block4c_se_squeeze = layers.GlobalAveragePooling2D()(block4c_bn)
        block4c_se_reshape = layers.Reshape((1, 1, 480))(block4c_se_squeeze)
        block4c_se_reduce = layers.Conv2D(20, 1, padding = 'SAME',activation="relu")(block4c_se_reshape)
        block4c_se_expand = layers.Conv2D(480, 1, padding = 'SAME', activation="relu")(block4c_se_reduce)
        block4c_se_excite = layers.Multiply()([block4c_bn, block4c_se_expand])


        block4c_project_conv = layers.Conv2D(80, 3, padding = 'SAME',activation="relu")(block4c_se_excite)
        block4c_project_bn = layers.BatchNormalization()(block4c_project_conv)
        # delte dropout
        block5a_expand_conv = layers.Conv2D(480, 1, padding = 'SAME',activation="relu")(block4c_project_bn)
        block5a_expand_bn = layers.BatchNormalization()(block5a_expand_conv)
        block5a_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block5a_expand_bn)
        block5a_bn = layers.BatchNormalization()(block5a_dwconv)
        block5a_se_squeeze = layers.GlobalAveragePooling2D()(block5a_bn)
        block5a_se_reshape = layers.Reshape((1, 1, 480))(block5a_se_squeeze)
        block5a_se_reduce = layers.Conv2D(20, 1, padding = 'SAME',activation="relu")(block5a_se_reshape)
        block5a_se_expand = layers.Conv2D(480, 1, padding = 'SAME', activation="relu")(block5a_se_reduce)
        block5a_se_excite = layers.Multiply()([block5a_bn, block5a_se_expand])


        block5a_project_conv = layers.Conv2D(112, 3, padding = 'SAME',activation="relu")(block5a_se_excite)
        block5a_project_bn = layers.BatchNormalization()(block5a_project_conv)
        block5b_expand_conv = layers.Conv2D(672, 1, padding = 'SAME',activation="relu")(block5a_project_bn)
        block5b_expand_bn = layers.BatchNormalization()(block5b_expand_conv)
        block5b_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block5b_expand_bn)
        block5b_bn = layers.BatchNormalization()(block5b_dwconv)
        block5b_se_squeeze = layers.GlobalAveragePooling2D()(block5b_bn)
        block5b_se_reshape = layers.Reshape((1, 1, 672))(block5b_se_squeeze)
        block5b_se_reduce = layers.Conv2D(28, 1, padding = 'SAME',activation="relu")(block5b_se_reshape)
        block5b_se_expand = layers.Conv2D(672, 1, padding = 'SAME', activation="relu")(block5b_se_reduce)
        block5b_se_excite = layers.Multiply()([block5b_bn, block5b_se_expand])


        block5b_project_conv = layers.Conv2D(112, 3, padding = 'SAME',activation="relu")(block5b_se_excite)
        block5b_project_bn = layers.BatchNormalization()(block5b_project_conv)
        # delte dropout
        block5c_expand_conv = layers.Conv2D(672, 1, padding = 'SAME',activation="relu")(block5b_project_bn)
        block5c_expand_bn = layers.BatchNormalization()(block5c_expand_conv)
        block5c_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block5c_expand_bn)
        block5c_bn = layers.BatchNormalization()(block5c_dwconv)
        block5c_se_squeeze = layers.GlobalAveragePooling2D()(block5c_bn)
        block5c_se_reshape = layers.Reshape((1, 1, 672))(block5c_se_squeeze)
        block5c_se_reduce = layers.Conv2D(28, 1, padding = 'SAME',activation="relu")(block5c_se_reshape)
        block5c_se_expand = layers.Conv2D(672, 1, padding = 'SAME', activation="relu")(block5c_se_reduce)
        block5c_se_excite = layers.Multiply()([block5c_bn, block5c_se_expand])


        block5c_project_conv = layers.Conv2D(112, 3, padding = 'SAME',activation="relu")(block5c_se_excite)
        block5c_project_bn = layers.BatchNormalization()(block5c_project_conv)

        top_conv = layers.Conv2D(1152, 1, padding = 'SAME',activation="relu")(block5c_project_bn)
        top_bn = layers.BatchNormalization()(top_conv)
        # x = layers.MaxPooling2D((2, 2), padding='same')(top_conv)
        # x = layers.Flatten()(x)
        avg_pool = layers.GlobalAveragePooling2D()(top_bn)
        prob = layers.Dense(units=int(feedback_bits/B), activation='sigmoid')(avg_pool)
        encoder_output = QuantizationLayer(B)(prob)
    return encoder_output
    
def Decoder(x,feedback_bits):
    B=4
    decoder_input = DeuantizationLayer(B)(x)
    # x = tf.keras.layers.Reshape((-1, int(feedback_bits/B)))(decoder_input)
    x = tf.reshape(decoder_input, (-1, int(feedback_bits/B)))
    x = layers.Dense(1024, activation='sigmoid')(x)
    # x = layers.Reshape((16, 32, 2))(x)
    # x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    # x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    # x_ini = layers.Reshape((16, 32, 2))(x)
    # x = layers.Conv2D(128, 3, padding = 'SAME',activation="relu")(x_ini)
    # x = layers.Conv2D(64, 3, padding = 'SAME',activation="relu")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
    # x_ini = keras.layers.Add()([x_ini, x])
    x_ini = layers.Reshape((16, 32, 2))(x)
    # feature map is (8, 16,)
    top_conv = layers.Conv2D(32, 3, padding = 'SAME', activation="relu")(x_ini)
    top_bn = layers.BatchNormalization()(top_conv)
    block5c_project_conv = layers.Conv2D(112, 3, padding = 'SAME',activation="relu")(top_bn)
    block5c_project_bn = layers.BatchNormalization()(block5c_project_conv)

    block5b_project_conv = layers.Conv2D(112, 3, padding = 'SAME',activation="relu")(block5c_project_bn)
    block5b_project_bn = layers.BatchNormalization()(block5b_project_conv)
    # delte dropout
    block5c_expand_conv = layers.Conv2D(672, 1, padding = 'SAME',activation="relu")(block5b_project_bn)
    block5c_expand_bn = layers.BatchNormalization()(block5c_expand_conv)
    # feature map is (4, 8,)
    block5c_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block5c_expand_bn)
    block5c_bn = layers.BatchNormalization()(block5c_dwconv)
    block5c_se_squeeze = layers.GlobalAveragePooling2D()(block5c_bn)
    block5c_se_reshape = layers.Reshape((1, 1, 672))(block5c_se_squeeze)
    block5c_se_reduce = layers.Conv2D(28, 1, padding = 'SAME',activation="relu")(block5c_se_reshape)
    block5c_se_expand = layers.Conv2D(672, 1, padding = 'SAME', activation="relu")(block5c_se_reduce)
    block5c_se_excite = layers.Multiply()([block5c_bn, block5c_se_expand])


    block5a_project_conv = layers.Conv2D(112, 3, padding = 'SAME',activation="relu")(block5c_se_excite)
    block5a_project_bn = layers.BatchNormalization()(block5a_project_conv)
    block5b_expand_conv = layers.Conv2D(672, 1, padding = 'SAME',activation="relu")(block5a_project_bn)
    block5b_expand_bn = layers.BatchNormalization()(block5b_expand_conv)
    # feature map is (2, 4,)
    block5b_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block5b_expand_bn)
    block5b_bn = layers.BatchNormalization()(block5b_dwconv)
    block5b_se_squeeze = layers.GlobalAveragePooling2D()(block5b_bn)
    block5b_se_reshape = layers.Reshape((1, 1, 672))(block5b_se_squeeze)
    block5b_se_reduce = layers.Conv2D(28, 1, padding = 'SAME',activation="relu")(block5b_se_reshape)
    block5b_se_expand = layers.Conv2D(672, 1, padding = 'SAME', activation="relu")(block5b_se_reduce)
    block5b_se_excite = layers.Multiply()([block5b_bn, block5b_se_expand])  


    block4c_project_conv = layers.Conv2D(80, 3, padding = 'SAME',activation="relu")(block5b_se_excite)
    block4c_project_bn = layers.BatchNormalization()(block4c_project_conv)
    # delte dropout
    block5a_expand_conv = layers.Conv2D(480, 1, padding = 'SAME',activation="relu")(block4c_project_bn)
    block5a_expand_bn = layers.BatchNormalization()(block5a_expand_conv)
    block5a_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block5a_expand_bn)
    block5a_bn = layers.BatchNormalization()(block5a_dwconv)
    block5a_se_squeeze = layers.GlobalAveragePooling2D()(block5a_bn)
    block5a_se_reshape = layers.Reshape((1, 1, 480))(block5a_se_squeeze)
    block5a_se_reduce = layers.Conv2D(20, 1, padding = 'SAME',activation="relu")(block5a_se_reshape)
    block5a_se_expand = layers.Conv2D(480, 1, padding = 'SAME', activation="relu")(block5a_se_reduce)
    block5a_se_excite = layers.Multiply()([block5a_bn, block5a_se_expand])


    block4b_project_conv = layers.Conv2D(80, 3, padding = 'SAME',activation="relu")(block5a_se_excite)
    block4b_project_bn = layers.BatchNormalization()(block4b_project_conv)
    # delte dropout
    block4c_expand_conv = layers.Conv2D(480, 1, padding = 'SAME',activation="relu")(block4b_project_bn)
    block4c_expand_bn = layers.BatchNormalization()(block4c_expand_conv)
    block4c_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block4c_expand_bn)
    block4c_bn = layers.BatchNormalization()(block4c_dwconv)
    block4c_se_squeeze = layers.GlobalAveragePooling2D()(block4c_bn)
    block4c_se_reshape = layers.Reshape((1, 1, 480))(block4c_se_squeeze)
    block4c_se_reduce = layers.Conv2D(20, 1, padding = 'SAME',activation="relu")(block4c_se_reshape)
    block4c_se_expand = layers.Conv2D(480, 1, padding = 'SAME', activation="relu")(block4c_se_reduce)
    block4c_se_excite = layers.Multiply()([block4c_bn, block4c_se_expand])


    block4a_project_conv = layers.Conv2D(80, 3, padding = 'SAME',activation="relu")(block4c_se_excite)
    block4a_project_bn = layers.BatchNormalization()(block4a_project_conv)
    block4b_expand_conv = layers.Conv2D(480, 1, padding = 'SAME',activation="relu")(block4a_project_bn)
    block4b_expand_bn = layers.BatchNormalization()(block4b_expand_conv)
    block4b_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block4b_expand_bn)
    block4b_bn = layers.BatchNormalization()(block4b_dwconv)
    block4b_se_squeeze = layers.GlobalAveragePooling2D()(block4b_bn)
    block4b_se_reshape = layers.Reshape((1, 1, 480))(block4b_se_squeeze)
    block4b_se_reduce = layers.Conv2D(20, 1, padding = 'SAME',activation="relu")(block4b_se_reshape)
    block4b_se_expand = layers.Conv2D(480, 1, padding = 'SAME', activation="relu")(block4b_se_reduce)
    block4b_se_excite = layers.Multiply()([block4b_bn, block4b_se_expand])


    block3b_project_conv = layers.Conv2D(40, 3, padding = 'SAME',activation="relu")(block4b_se_excite)
    block3b_project_bn = layers.BatchNormalization()(block3b_project_conv)
    # delte dropout
    block4a_expand_conv = layers.Conv2D(240, 1, padding = 'SAME',activation="relu")(block3b_project_bn)
    block4a_expand_bn = layers.BatchNormalization()(block4a_expand_conv)  


    block3a_project_conv = layers.Conv2D(40, 3, padding = 'SAME',activation="relu")(block4a_expand_bn)
    block3a_project_bn = layers.BatchNormalization()(block3a_project_conv)
    block3b_expand_conv = layers.Conv2D(240, 1, padding = 'SAME',activation="relu")(block3a_project_bn)
    block3b_expand_bn = layers.BatchNormalization()(block3b_expand_conv)
    block3b_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block3b_expand_bn)
    block3b_bn = layers.BatchNormalization()(block3b_dwconv)
    block3b_se_squeeze = layers.GlobalAveragePooling2D()(block3b_bn)
    block3b_se_reshape = layers.Reshape((1, 1, 240))(block3b_se_squeeze)
    block3b_se_reduce = layers.Conv2D(10, 1, padding = 'SAME',activation="relu")(block3b_se_reshape)
    block3b_se_expand = layers.Conv2D(240, 1, padding = 'SAME', activation="relu")(block3b_se_reduce)
    block3b_se_excite = layers.Multiply()([block3b_bn, block3b_se_expand])


    block2b_project_conv = layers.Conv2D(24, 3, padding = 'SAME',activation="relu")(block3b_se_excite)
    block2b_project_bn = layers.BatchNormalization()(block2b_project_conv)
    # delte dropout
    block3a_expand_conv = layers.Conv2D(144, 1, padding = 'SAME',activation="relu")(block2b_project_bn)
    block3a_expand_bn = layers.BatchNormalization()(block3a_expand_conv)
    block3a_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block3a_expand_bn)
    block3a_bn = layers.BatchNormalization()(block3a_dwconv)
    block3a_se_squeeze = layers.GlobalAveragePooling2D()(block3a_bn)
    block3a_se_reshape = layers.Reshape((1, 1, 144))(block3a_se_squeeze)
    block3a_se_reduce = layers.Conv2D(6, 1, padding = 'SAME', activation="relu")(block3a_se_reshape)
    block3a_se_expand = layers.Conv2D(144, 1, padding = 'SAME', activation="relu")(block3a_se_reduce)
    block3a_se_excite = layers.Multiply()([block3a_bn, block3a_se_expand]) 


    block2a_project_conv = layers.Conv2D(24, 3, padding = 'SAME',activation="relu")(block3a_se_excite)
    block2a_project_bn = layers.BatchNormalization()(block2a_project_conv)
    # feature map is (4, 8,)
    # block2b_project_up = layers.UpSampling2D((2,2))(block2a_project_bn)
    block2b_expand_conv = layers.Conv2D(144, 1, padding = 'SAME',activation="relu")(block2a_project_bn)
    block2b_expand_bn = layers.BatchNormalization()(block2b_expand_conv)
    block2b_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block2b_expand_bn)
    block2b_bn = layers.BatchNormalization()(block2b_dwconv)
    block2b_se_squeeze = layers.GlobalAveragePooling2D()(block2b_bn)
    block2b_se_reshape = layers.Reshape((1, 1, 144))(block2b_se_squeeze)
    block2b_se_reduce = layers.Conv2D(6, 1, padding = 'SAME', activation="relu")(block2b_se_reshape)
    block2b_se_expand = layers.Conv2D(144, 1, padding = 'SAME', activation="relu")(block2b_se_reduce)
    block2b_se_excite = layers.Multiply()([block2b_bn, block2b_se_expand])  


    block1a_project_conv = layers.Conv2D(16, 3, padding = 'SAME',activation="relu")(block2b_se_excite)
    block1a_project_bn = layers.BatchNormalization()(block1a_project_conv)
    # feature map is (8, 16,)
    # block1a_project_up = layers.UpSampling2D((2,2))(block1a_project_bn)
    block2a_expand_conv = layers.Conv2D(96, 1, padding = 'SAME',activation="relu")(block1a_project_bn)
    block2a_expand_bn = layers.BatchNormalization()(block2a_expand_conv)
    block2a_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(block2a_expand_bn)
    block2a_bn = layers.BatchNormalization()(block2a_dwconv)
    block2a_se_squeeze = layers.GlobalAveragePooling2D()(block2a_bn)
    block2a_se_reshape = layers.Reshape((1, 1, 96))(block2a_se_squeeze)
    block2a_se_reduce = layers.Conv2D(4, 1, padding = 'SAME', activation="relu")(block2a_se_reshape)
    block2a_se_expand = layers.Conv2D(96, 1, padding = 'SAME', activation="relu")(block2a_se_reduce)
    block2a_se_excite = layers.Multiply()([block2a_bn, block2a_se_expand])   


    stem_conv = layers.Conv2D(32, 3, padding = 'SAME', activation="relu")(block2a_se_excite)
    stem_bn = layers.BatchNormalization()(stem_conv)
    # feature map is (16, 32,)
    # stem_up = layers.UpSampling2D((2,2))(stem_bn)
    blockla_dwconv = layers.DepthwiseConv2D(1, padding = 'SAME', activation="relu")(stem_bn)
    blockla_bn = layers.BatchNormalization()(blockla_dwconv)
    block1a_se_squeeze = layers.GlobalAveragePooling2D()(blockla_bn)
    block1a_se_reshape = layers.Reshape((1, 1, 32))(block1a_se_squeeze)
    block1a_se_reduce = layers.Conv2D(8, 1, padding = 'SAME', activation="relu")(block1a_se_reshape)
    block1a_se_expand = layers.Conv2D(32, 1, padding = 'SAME', activation="relu")(block1a_se_reduce)
    block1a_se_excite = layers.Multiply()([blockla_bn, block1a_se_expand])          


    decoder_output = layers.Conv2D(2, 3, padding = 'SAME',activation="sigmoid")(block1a_se_excite)

    return decoder_output


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def Score(NMSE):
    score = 1 - NMSE
    return score

# Return keywords of your own custom layers to ensure that model
# can be successfully loaded in test file.
def get_custom_objects():
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer}

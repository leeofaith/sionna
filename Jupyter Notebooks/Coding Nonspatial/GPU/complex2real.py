import numpy as np
import tensorflow as tf
# For the implementation of the Keras models
from tensorflow import keras
from keras.layers import Layer

class Complex2Real(tf.keras.layers.Layer):

    def __init__(self,num_rx_ant,num_tx_ant):

        self.NUM_RX_ANT = num_rx_ant
        self.NUM_TX_ANT = num_tx_ant

    def C2R(self,X,H,Y):
        
        BATCH_SIZE = H.shape[0]
        NUM_RX_ANT = self.NUM_RX_ANT
        NUM_TX_ANT = self.NUM_TX_ANT

        # print('BATCH_SIZE =',BATCH_SIZE)
        # print('X shape =',tf.shape(X))
        # print('X =',X)
        # print('H shape =',tf.shape(H))
        # print('H =',H)
        H_shape = tf.shape(H)
        # print('H_shape =',H_shape)
        # print('Y shape =',tf.shape(Y))
        # print('Y =',Y)

        X_inCH = tf.expand_dims(X, axis=-1)
        # print('X_inCH =',X_inCH)
        X_inCH_real_part = tf.math.real(X_inCH);
        # print('X_reshaped_real_part =',X_reshaped_real_part)
        X_inCH_imag_part = tf.math.imag(X_inCH);
        # print('X_imag_part =',X_reshaped_imag_part)
        X_inCH_tmp_real = tf.concat([X_inCH_real_part, X_inCH_imag_part], axis=2)
        # print('X_tmp_real =',X_tmp_real)
        e = tf.reshape(X_inCH_tmp_real,[BATCH_SIZE,NUM_TX_ANT,2,1]) # [BATCH_SIZE,NUM_TX_ANT,2,1] = [1,-1,2,1]
        # print('e =',e)
        f = tf.reshape(tf.repeat(e, repeats=NUM_RX_ANT, axis=0),[BATCH_SIZE,NUM_RX_ANT,NUM_TX_ANT,2,1])
        # print('f =',f)
        X_inCH_real = f

        # print('X_inCH_real shape =',tf.shape(X_inCH_real))
        # print('X_inCH_real =',X_inCH_real)

        Y_real_part = tf.math.real(Y);
        # print('Y_real_part =',Y_real_part)
        Y_imag_part = tf.math.imag(Y);
        # print('Y_imag_part =',Y_imag_part)
        Y_real_part_reshaped = tf.reshape(Y_real_part,[BATCH_SIZE,NUM_RX_ANT,1,1])
        # print('Y_real_part_reshaped =',Y_real_part_reshaped)
        Y_imag_part_reshaped = tf.reshape(Y_imag_part,[BATCH_SIZE,NUM_RX_ANT,1,1])
        # print('Y_imag_part_reshaped =',Y_imag_part_reshaped)
        Y_real = tf.concat([Y_real_part_reshaped, Y_imag_part_reshaped], axis=2)
        
        # print('Y_real shape =',tf.shape(Y_real))
        # print('Y_real =',Y_real)


        # H_Reshaped_real = tf.concat([tf.concat([H_real_part_Reshaped, -H_imag_part_Reshaped], axis=2), tf.concat([H_imag_part_Reshaped, H_real_part_Reshaped], axis=2)],axis=1)
        H_Reshaped = tf.reshape(H,[BATCH_SIZE,-1,1])
        # print('H_Reshaped =',H_Reshaped)
        H_Reshaped_real_part = tf.math.real(H_Reshaped);
        # print('H_Reshaped_real_part =',H_Reshaped_real_part)
        H_Reshaped_imag_part = tf.math.imag(H_Reshaped);
        # print('H_Reshaped_imag_part =',H_Reshaped_imag_part);

        H_real_part_Reshaped = tf.reshape(H_Reshaped_real_part,[BATCH_SIZE,-1,1])
        # print('H_real_part_Reshaped =',H_real_part_Reshaped)
        H_imag_part_Reshaped = tf.reshape(H_Reshaped_imag_part,[BATCH_SIZE,-1,1])
        # print('H_imag_part_Reshaped =',H_imag_part_Reshaped)

        b = tf.concat([H_real_part_Reshaped, -H_imag_part_Reshaped], axis=2)
        # print('b =',b)
        c = tf.concat([H_imag_part_Reshaped, H_real_part_Reshaped], axis=2)
        # print('c =',c)
        b_trans = tf.reshape(b, [BATCH_SIZE, NUM_RX_ANT*NUM_TX_ANT, -1, 2])
        # print('b_trans =',b_trans)
        c_trans = tf.reshape(c, [BATCH_SIZE, NUM_RX_ANT*NUM_TX_ANT, -1, 2])
        # print('c_trans =',c_trans)
        d = tf.concat([b_trans, c_trans], axis=2)
        # print('d =',d)
        H_real = tf.reshape(d,[BATCH_SIZE,NUM_RX_ANT,NUM_TX_ANT,2,2])
       
        # print('H_real shape =',tf.shape(H_real))
        # print('H_real =',H_real)
        
        # Y = tf.matmul(H, X)
        # print('Y =',Y)
        # result = tf.matmul(H_real, X_real)
        # # print('result =',result)
        # sum = tf.reduce_sum(result, axis=2)
        # # print('sum =',sum)
        # sum_real,sum_imag = tf.split(sum, num_or_size_splits=2, axis=2)
        # # print('sum_real =',sum_real)
        # # print('sum_imag =',sum_imag)
        # sum_complex = tf.squeeze(tf.complex(sum_real,sum_imag),axis=-1)
        # print('sum_complex =',sum_complex)
        # if tf.reduce_all(tf.equal(Y,sum_complex)):
        #     print('Y = sum_complex!!!')

        return X_inCH_real,H_real,Y_real
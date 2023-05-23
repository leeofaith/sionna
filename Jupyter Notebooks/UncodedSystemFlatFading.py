import tensorflow as tf
# For the implementation of the Keras models
from tensorflow import keras
from keras import Model
# for performance measurements
import time

# GPU Configuration and Imports
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

import matplotlib.pyplot as plt
import numpy as np
import sys

# Import Sionna
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna as sn

from sionna.mimo import lmmse_equalizer, zf_equalizer
from xyDIP import DeepImagePrior
from complex2real import Complex2Real

class UncodedSystemFlatFading(Model): # Inherits from Keras Model
    def __init__(self,
                 Block_Length,
                 NUM_BITS_PER_SYMBOL,
                 CONSTELLATION_TYPE,
                 DEMAPPING_METHOD,
                 NUM_TX_ANT,
                 NUM_RX_ANT):
        """
        A keras model of an uncoded transmission over the AWGN channel.

        Parameters
        ----------
        NUM_BITS_PER_SYMBOL: int
            The number of bits per constellation symbol, e.g., 4 for QAM16.

        Block_Length: int
            The number of bits per transmitted message block (will be the codeword length later).

        Input
        -----
        BATCH_SIZE: int
            The BATCH_SIZE of the Monte-Carlo simulation.

        EBN0_DB: float
            The `Eb/No` value (=rate-adjusted SNR) in dB.

        Output
        ------
        (bits, llr):
            Tuple:

        bits: tf.float32
            A tensor of shape `[BATCH_SIZE, Block_Length] of 0s and 1s
            containing the transmitted information bits.

        llr: tf.float32
            A tensor of shape `[BATCH_SIZE, Block_Length] containing the
            received log-likelihood-ratio (LLR) values.
        """

        super().__init__() # Must call the Keras model initializer
        
        self.Block_Length = Block_Length
        self.NUM_BITS_PER_SYMBOL = NUM_BITS_PER_SYMBOL
        self.CONSTELLATION_TYPE = CONSTELLATION_TYPE
        self.DEMAPPING_METHOD = DEMAPPING_METHOD
        self.NUM_TX_ANT = NUM_TX_ANT
        self.NUM_RX_ANT = NUM_RX_ANT
        self.constellation = sn.mapping.Constellation(constellation_type=self.CONSTELLATION_TYPE,
                                                      num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL)
        self.mapper = sn.mapping.Mapper(constellation_type=self.CONSTELLATION_TYPE,
                                        num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL)
        self.demapper = sn.mapping.Demapper(demapping_method=self.DEMAPPING_METHOD,
                                            constellation_type=self.CONSTELLATION_TYPE,
                                            num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL)
        self.SymbolDemapper = sn.mapping.SymbolDemapper(constellation_type=self.CONSTELLATION_TYPE,
                                                        num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
                                                        hard_out=True)
        # self.LDPC5GEncoder = sn.fec.ldpc.encoding.LDPC5GEncoder()
        # self.LDPC5GDecoder = sn.fec.ldpc.decoding.LDPC5GDecoder()
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        self.flatfading_channel = sn.channel.FlatFadingChannel(num_tx_ant=self.NUM_TX_ANT,
                                                               num_rx_ant=self.NUM_TX_ANT,
                                                               add_awgn=True,
                                                               return_channel=True)
        # self.KroneckerModel = sn.channel.KroneckerModel()
        # self.exp_corr_mat = sn.channel.utils.exp_corr_mat()
        # self.hard_decisions = sn.utils.misc.hard_decisions()
        self.dip = DeepImagePrior(num_rx_ant=NUM_RX_ANT,
                                  num_tx_ant=NUM_TX_ANT, # Number of transmitted symbol in real domain
                                  M=2**NUM_BITS_PER_SYMBOL, # Modulation order (16), 4 for 4QAM, 16 for 16QAM(4 bits/symbol)
                                  iteration=100, # Number of max iterations used for DIP: 100
                                  LR=0.01, # Learning rate,  typically set to 0.01
                                  buffer_size=30, # Iterations stored,  typically set to 30
                                  threshold=0.001, # Threshold of DIP stop,, typically set to 0.001
                                  stop=True) # True
        self.c2r = Complex2Real(num_rx_ant=NUM_RX_ANT,num_tx_ant=NUM_TX_ANT)

    # @tf.function # Enable graph execution to speed things up
    def __call__(self,
                 NUM_DATA_GROUP,
                 BATCH_SIZE,
                 EBN0_DB_MIN,
                 EBN0_DB_MAX,
                 NUM_EBN0_POINTS):
        
        tf.config.run_functions_eagerly(True)
        snrs = np.linspace(EBN0_DB_MIN,EBN0_DB_MAX,NUM_EBN0_POINTS)
        # bers = []
        sers_zf = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        sers_lmmse = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        sers_dip = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))

        for i in range(0, NUM_DATA_GROUP):
            print('|                   Data Group {}                     |'.format(i+1))
            print('______________________________________________________')
            print('|  EBN0[DB]  |  SER(ZF)  |  SER(LMMSE)  |  SER(DIP)  |')
            print('______________________________________________________')
            b = self.binary_source([BATCH_SIZE,self.NUM_TX_ANT,self.Block_Length])
            x =  self.mapper(b)
            shape = tf.shape(x)
            x_reshape = tf.reshape(x,[-1, self.NUM_TX_ANT])
            j = 0

            for EBN0_DB in np.linspace(EBN0_DB_MIN,EBN0_DB_MAX,NUM_EBN0_POINTS):
                print('            ')
                print("{:^12.8f}".format(EBN0_DB), end='  ')
                # no channel coding used; we set coderate=1.0
                no = sn.utils.ebnodb2no(ebno_db=EBN0_DB,
                                        num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
                                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here
                x_ind = self.SymbolDemapper([x, no])

                # y and h are the Channel Output and Channel Realizations, respectively
                y, h = self.flatfading_channel([x_reshape, no])
                s = tf.cast(no*tf.eye(self.NUM_RX_ANT, self.NUM_RX_ANT), y.dtype)
                x_hat_zf, no_eff_zf = zf_equalizer(y, h, s)
                x_hat_lmmse, no_eff_lmmse = lmmse_equalizer(y, h, s)
                X_inCH_real,H_real,Y_real = self.c2r.C2R(x_reshape,h,y)
                x_dip_ay,num_stop_point = self.dip.DIP(Y_real,H_real)
                x_dip_ay_real_part,x_dip_ay_imag_part = tf.split(x_dip_ay, num_or_size_splits=2, axis=2)
                
                x_hat_dip = tf.squeeze(tf.squeeze(tf.complex(x_dip_ay_real_part,x_dip_ay_imag_part),axis=-1),axis=-1)
                x_hat_zf = tf.reshape(x_hat_zf, shape)
                x_hat_lmmse = tf.reshape(x_hat_lmmse, shape)
                x_hat_dip = tf.reshape(x_hat_dip, shape)

                # no_eff_zf = tf.reshape(no_eff_zf, shape)
                # no_eff_lmmse = tf.reshape(no_eff_lmmse, shape)

                # llr_zf = self.demapper([x_hat_zf, no_eff_zf])
                # b_hat_zf = self.LDPC5GDecoder(llr_zf)

                x_ind_hat_zf = self.SymbolDemapper([x_hat_zf, no])
                x_ind_hat_lmmse = self.SymbolDemapper([x_hat_lmmse, no])
                x_ind_hat_dip = self.SymbolDemapper([tf.cast(x_hat_dip, dtype=tf.complex64), no])

                ser_zf = sn.utils.compute_ser(x_ind, x_ind_hat_zf)
                ser_lmmse = sn.utils.compute_ser(x_ind, x_ind_hat_lmmse)
                ser_dip = sn.utils.compute_ser(x_ind, x_ind_hat_dip)
                print("{:^12.8f}".format(ser_zf), end='  ')
                print("{:^12.8f}".format(ser_lmmse), end='  ')
                print("{:^12.8f}".format(ser_lmmse), end='  ')
                print("\n")
                print('______________________________________________________')
                sers_zf[i, j] = ser_zf
                sers_lmmse[i, j] = ser_lmmse
                sers_dip[i, j] = ser_dip
                j = j+1
            print('Done')

            plt.figure(1)
            plt.axes().set_aspect(1)
            plt.grid(True)
            plt.title('Flat-Fading Channel Constellation', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel('REAL', fontsize=10)
            plt.ylabel('IMAG', fontsize=10)
            plt.scatter(tf.math.real(x), tf.math.imag(x), s=16, c='b', label='TX')
            plt.scatter(tf.math.real(x_hat_zf), tf.math.imag(x_hat_zf), s=16, c='y', label='ZF')
            plt.scatter(tf.math.real(x_hat_lmmse), tf.math.imag(x_hat_lmmse), s=16, c='g', label='LMMSE')
            plt.scatter(tf.math.real(x_hat_dip), tf.math.imag(x_hat_dip), s=16, c='r', label='DIP')
            plt.legend(loc='lower left', fontsize=8)
            plt.tight_layout()

            plt.figure(2)
            title = "SER: Noncoding MIMO Falt-Fading with ZF, MMSE, DIP Equalizer"
            xlabel = "$E_b/N_0$ (dB)"
            ylabel = "SER (log)"
            plt.title(title, fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel(xlabel, fontsize=10)
            plt.ylabel(ylabel, fontsize=10)
            plt.grid(which="both")
            plt.semilogy(snrs, x_hat_zf, 'b', label='ZF')
            plt.semilogy(snrs, x_hat_lmmse, 'g', label='LMMSE')
            plt.semilogy(snrs, x_hat_dip, 'r', label='DIP')
            plt.legend(loc='lower left', fontsize=8)
            plt.tight_layout()

            plt.show()

        sers_zf_mean = np.mean(sers_zf, axis=0)
        sers_lmmse_mean = np.mean(sers_lmmse, axis=0)
        sers_dip_mean = np.mean(sers_dip, axis=0)

        plt.figure(1)
        plt.axes().set_aspect(1)
        plt.grid(True)
        plt.title('Flat-Fading Channel Constellation', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('REAL', fontsize=10)
        plt.ylabel('IMAG', fontsize=10)
        plt.scatter(tf.math.real(x), tf.math.imag(x), s=16, c='b', label='TX')
        plt.scatter(tf.math.real(x_hat_zf), tf.math.imag(x_hat_zf), s=16, c='y', label='ZF')
        plt.scatter(tf.math.real(x_hat_lmmse), tf.math.imag(x_hat_lmmse), s=16, c='g', label='LMMSE')
        plt.scatter(tf.math.real(x_hat_dip), tf.math.imag(x_hat_dip), s=16, c='r', label='DIP')
        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()

        plt.figure(2)
        title = "SER: Noncoding MIMO Falt-Fading with ZF, MMSE, DIP Equalizer"
        xlabel = "$E_b/N_0$ (dB)"
        ylabel = "SER (log)"
        plt.title(title, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(which="both")
        plt.semilogy(snrs, sers_zf_mean, 'b', label='ZF')
        plt.semilogy(snrs, sers_lmmse_mean, 'g', label='LMMSE')
        plt.semilogy(snrs, sers_dip_mean, 'r', label='DIP')
        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()

        plt.show()

        return snrs, sers_zf_mean, sers_lmmse_mean, sers_dip_mean
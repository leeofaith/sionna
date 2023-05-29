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

# Import Plot Function
from bokeh.plotting import show
from PlotYCodeYCorr import plotycodeycorr

# Import Sionna
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna as sn

from sionna.mimo import lmmse_equalizer, zf_equalizer
from DIP import DeepImagePrior
from sionna.utils.misc import hard_decisions
from sionna.utils.metrics import compute_ber, count_errors
from sionna.mimo import complex2real_vector, complex2real_channel
from sionna.channel.utils import exp_corr_mat
from sionna.channel import KroneckerModel

class ycodeycorr(Model): # Inherits from Keras Model
    def __init__(self,
                 Block_Length,
                 NUM_BITS_PER_SYMBOL,
                 CONSTELLATION_TYPE,
                 DEMAPPING_METHOD,
                 NUM_RX_ANT,
                 NUM_TX_ANT,
                 CODERATE,
                 CORRELATION_INDEX_MIN,
                 CORRELATION_INDEX_MAX,
                 CORRELATION_INDEX_POINTS):
        
        super().__init__() # Must call the Keras model initializer
        
        self.Block_Length = Block_Length
        self.NUM_BITS_PER_SYMBOL = NUM_BITS_PER_SYMBOL
        self.CONSTELLATION_TYPE = CONSTELLATION_TYPE
        self.DEMAPPING_METHOD = DEMAPPING_METHOD
        self.NUM_RX_ANT = NUM_RX_ANT
        self.NUM_TX_ANT = NUM_TX_ANT
        self.CODERATE = CODERATE
        self.Code_Length = int(self.Block_Length/self.CODERATE)
        self.CORRELATION_INDEX_MIN = CORRELATION_INDEX_MIN
        self.CORRELATION_INDEX_MAX = CORRELATION_INDEX_MAX
        self.CORRELATION_INDEX_POINT = CORRELATION_INDEX_POINTS

        ### Constellation
        self.constellation = sn.mapping.Constellation(constellation_type=self.CONSTELLATION_TYPE,
                                                      num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL)
        ### Mapper
        self.mapper = sn.mapping.Mapper(constellation_type=self.CONSTELLATION_TYPE,
                                        num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
                                        return_indices=True)
        ### Demapper
        self.demapper = sn.mapping.Demapper(demapping_method=self.DEMAPPING_METHOD,
                                            constellation_type=self.CONSTELLATION_TYPE,
                                            num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL)
        self.SymbolDemapper = sn.mapping.SymbolDemapper(constellation_type=self.CONSTELLATION_TYPE,
                                                        num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
                                                        hard_out=True)
        ### LDPC Encoder
        ## Block_Length: k, Code_Length: n
        self.LDPC5GEncoder = sn.fec.ldpc.encoding.LDPC5GEncoder(self.Block_Length,self.Code_Length)
        ### LDPC Decoder
        self.LDPC5GDecoder = sn.fec.ldpc.decoding.LDPC5GDecoder(self.LDPC5GEncoder, hard_out=True)
        ### Binary Source
        self.binary_source = sn.utils.BinarySource()
        ### AWGN Channel
        self.awgn_channel = sn.channel.AWGN()
        ### Flat Fading Channel
        self.flatfading_channel = sn.channel.FlatFadingChannel(num_tx_ant=self.NUM_TX_ANT,
                                                               num_rx_ant=self.NUM_RX_ANT,
                                                               add_awgn=True,
                                                               return_channel=True)
        ### DIP Equalizer
        self.dip = DeepImagePrior(user_num=2*NUM_TX_ANT,
                                  M=2**NUM_BITS_PER_SYMBOL,
                                  iteration=100,
                                  LR=0.008,
                                  buffer_size=30,
                                  threshold=0.0005,
                                  stop=True)

    def __call__(self,
                 NUM_DATA_GROUP,
                 BATCH_SIZE,
                 EBN0_DB_MIN,
                 EBN0_DB_MAX,
                 NUM_EBN0_POINTS):
        
        # tf.config.run_functions_eagerly(True)

        ### Number of Spatial Correlation Group
        NUM_COR_GROUP = (self.CORRELATION_INDEX_POINT)*(self.CORRELATION_INDEX_POINT)
        COR_GROUP =  ([0.1,0.1],[0.1,0.5],[0.1,0.9],[0.5,0.1],[0.5,0.5],[0.5,0.9],[0.9,0.1],[0.9,0.5],[0.9,0.9])

        snrs = np.linspace(EBN0_DB_MIN,EBN0_DB_MAX,NUM_EBN0_POINTS)
        ### Uncode Variables Definition
        # sers_zf = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        # sers_lmmse = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        # sers_dip = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        # bers_zf = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        # bers_lmmse = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        # bers_dip = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))

        ### Code Variables Definition

        coded_sers_zf = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        coded_sers_lmmse = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        coded_sers_dip = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        coded_bers_zf = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        coded_bers_lmmse = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        coded_bers_dip = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))

        coded_sers_zf_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        coded_sers_lmmse_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        coded_sers_dip_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        coded_bers_zf_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        coded_bers_lmmse_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        coded_bers_dip_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))

        ### Spatial Correlation Loop Variable m
        m = 0

        for TX_ANT_CORRELATION in np.linspace(self.CORRELATION_INDEX_MIN,
                                                    self.CORRELATION_INDEX_MAX,
                                                    self.CORRELATION_INDEX_POINT):
            
            ### Assign Transmit Antenna Correlation
            tx_correlation = exp_corr_mat(TX_ANT_CORRELATION, self.NUM_TX_ANT)
            
            for RX_ANT_CORRELATION in np.linspace(self.CORRELATION_INDEX_MIN,
                                                        self.CORRELATION_INDEX_MAX,
                                                        self.CORRELATION_INDEX_POINT):
                
                ### Assign Receive Antenna Correlation
                rx_correlation = exp_corr_mat(RX_ANT_CORRELATION, self.NUM_RX_ANT)
                
                ### Applye Antenna Spatial Correlation to Channel 
                self.flatfading_channel.spatial_corr = KroneckerModel(tx_correlation, rx_correlation)
                
                ### Data Group Loop Variable i
                i = 0
                print('|                                                         TX_ANT_CORRELATION: {} & RX_ANT_CORRELATION: {}                                                         |'.format(TX_ANT_CORRELATION,RX_ANT_CORRELATION))

                for i in range(0, NUM_DATA_GROUP):
                    print('|                                                                            Data Group {}                                                                           |'.format(i+1))
                    print('|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|')
                    print('|  EBN0[DB]  |  SER(ZF)  |  BER(ZF)  | Bit Errors(ZF) |  SER(LMMSE)  |  BER(LMMSE)  | Bit Errors(LMMSE) |  SER(DIP)  |  BER(DIP)  | Bit Errors(DIP) | Time Spent(s) |')
                    print('|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|')
                    b = self.binary_source([BATCH_SIZE,self.NUM_TX_ANT,self.Block_Length])

                    ### No Coding
                    # x, x_int_rep =  self.mapper(b)
                    # shape = tf.shape(x)
                    # x_reshape = tf.reshape(x,[-1, self.NUM_TX_ANT])

                    ### LDPC Encoding
                    coded_b = self.LDPC5GEncoder(b)
                    coded_x, coded_x_int_rep =  self.mapper(coded_b)
                    shape_coded_x = tf.shape(coded_x)
                    coded_x_reshape = tf.reshape(coded_x,[-1, self.NUM_TX_ANT])            
                    
                    j = 0

                    for EBN0_DB in np.linspace(EBN0_DB_MIN,EBN0_DB_MAX,NUM_EBN0_POINTS):
                        start_time = time.time()
                        print('|', end=' ')
                        print("{}|".format(EBN0_DB).rjust(12), end='')
                        # no channel coding used; we set coderate=1.0
                        no = sn.utils.ebnodb2no(ebno_db=EBN0_DB,
                                                num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
                                                coderate=self.CODERATE) # Coderate set to 1 as we do uncoded transmission here

                        # y and h are the Channel Output and Channel Realizations, respectively
                        ### Channel Transmission
                        ##  Uncoded
                        # y, h = self.flatfading_channel([x_reshape, no]) # type: ignore
                        ##  Coded
                        coded_y, coded_h = self.flatfading_channel([coded_x_reshape, no]) # type: ignore
                        ##  Same s for Uncoded and Coded Transmission
                        s = tf.cast(no*tf.eye(self.NUM_RX_ANT, self.NUM_RX_ANT), coded_y.dtype)

                        ### Deep Image Prior Eqaulizer
                        ##  Uncoded
                        ##  Complex to Real
                        # y_shape = tf.shape(y)
                        # y_reshape = tf.reshape(y,[-1, self.NUM_RX_ANT])
                        # y_reshape_real_part = np.real(y_reshape)
                        # y_reshape_image_part = np.imag(y_reshape)
                        # y_reshape_real = np.concatenate([y_reshape_real_part,y_reshape_image_part], axis=1)
                        # h_shape = tf.shape(h)
                        # h_reshape = tf.reshape(h,[-1, self.NUM_RX_ANT, self.NUM_TX_ANT])
                        # h_reshape_real_part = np.real(h_reshape)
                        # h_reshape_imag_part = np.imag(h_reshape)
                        # h_reshape_real = np.concatenate([np.concatenate([h_reshape_real_part, -h_reshape_imag_part], axis=2),
                        #                                  np.concatenate([h_reshape_imag_part, h_reshape_real_part], axis=2)],axis=1)
                        ##  DIP Equalizer
                        # x_dip_ay,num_stop_point = self.dip.DIP(y_reshape_real,h_reshape_real)
                        # x_dip_ay_real_part,x_dip_ay_imag_part = np.split(x_dip_ay, indices_or_sections=2, axis=1)
                        # x_hat_dip = tf.cast(tf.reshape(tf.complex(x_dip_ay_real_part,x_dip_ay_imag_part), shape), dtype=tf.complex64)
                        ##  Assume noise variance equal to channel noise
                        # no_eff_dip = no*np.ones(shape)

                        ### Deep Image Prior Eqaulizer
                        ##  Coded
                        ##  Complex to Real
                        coded_y_shape = tf.shape(coded_y)
                        coded_y_reshape = tf.reshape(coded_y,[-1, self.NUM_RX_ANT])
                        coded_y_reshape_real_part = tf.math.real(coded_y_reshape)
                        coded_y_reshape_image_part = tf.math.imag(coded_y_reshape)
                        coded_y_reshape_real = tf.concat([coded_y_reshape_real_part,coded_y_reshape_image_part], axis=1)
                        coded_h_shape = tf.shape(coded_h)
                        coded_h_reshape = tf.reshape(coded_h,[-1, self.NUM_RX_ANT, self.NUM_TX_ANT])
                        coded_h_reshape_real_part = tf.math.real(coded_h_reshape)
                        coded_h_reshape_imag_part = tf.math.imag(coded_h_reshape)
                        coded_h_reshape_real = tf.concat([tf.concat([coded_h_reshape_real_part, tf.multiply(coded_h_reshape_imag_part,-1)], axis=2),
                                                          tf.concat([coded_h_reshape_imag_part, coded_h_reshape_real_part], axis=2)],axis=1)
                        ##  DIP Equalizer
                        coded_x_dip_ay,num_stop_point = self.dip.DIP(np.array(coded_y_reshape_real),np.array(coded_h_reshape_real))
                        coded_x_dip_ay_real_part,coded_x_dip_ay_imag_part = np.split(coded_x_dip_ay, indices_or_sections=2, axis=1)
                        coded_x_hat_dip = tf.cast(tf.reshape(tf.complex(coded_x_dip_ay_real_part,coded_x_dip_ay_imag_part), shape_coded_x), dtype=tf.complex64)
                        ##  Assume noise variance equal to channel noise
                        coded_no_eff_dip = no*tf.ones(shape_coded_x)

                        ### Zero-Forcing Equalizer
                        ##  Uncoded
                        # x_hat_zf, no_eff_zf = zf_equalizer(y, h, s)
                        # no_eff_zf = tf.reshape(no_eff_zf, shape)
                        # x_hat_zf = tf.reshape(x_hat_zf, shape)

                        ##  Coded
                        coded_x_hat_zf, coded_no_eff_zf = zf_equalizer(coded_y, coded_h, s)
                        coded_no_eff_zf = tf.reshape(coded_no_eff_zf, shape_coded_x)
                        coded_x_hat_zf = tf.reshape(coded_x_hat_zf, shape_coded_x)

                        ### LMMSE Equalizer
                        ##  Uncoded
                        # x_hat_lmmse, no_eff_lmmse = lmmse_equalizer(y, h, s)
                        # no_eff_lmmse = tf.reshape(no_eff_lmmse, shape)
                        # x_hat_lmmse = tf.reshape(x_hat_lmmse, shape)

                        ### LMMSE Equalizer
                        ##  Coded
                        coded_x_hat_lmmse, coded_no_eff_lmmse = lmmse_equalizer(coded_y, coded_h, s)
                        coded_no_eff_lmmse = tf.reshape(coded_no_eff_lmmse, shape_coded_x)
                        coded_x_hat_lmmse = tf.reshape(coded_x_hat_lmmse, shape_coded_x)

                        ### Soft Decision Outputs Received Symbols(Integer)
                        ##  Uncoded
                        # x_int_zf = self.SymbolDemapper([x_hat_zf, no_eff_zf])
                        # x_int_lmmse = self.SymbolDemapper([x_hat_lmmse, no_eff_lmmse])
                        # x_int_dip = self.SymbolDemapper([x_hat_dip, no_eff_dip])
                        ##  Coded
                        coded_x_int_zf = self.SymbolDemapper([coded_x_hat_zf, coded_no_eff_zf])
                        coded_x_int_lmmse = self.SymbolDemapper([coded_x_hat_lmmse, coded_no_eff_lmmse])
                        coded_x_int_dip = self.SymbolDemapper([coded_x_hat_dip, coded_no_eff_dip])

                        ### SER Calculation
                        ##  Uncoded
                        # ser_zf = sn.utils.compute_ser(x_int_rep, x_int_zf)
                        # ser_lmmse = sn.utils.compute_ser(x_int_rep, x_int_lmmse)
                        # ser_dip = sn.utils.compute_ser(x_int_rep, x_int_dip)
                        ##  Coded
                        coded_ser_zf = sn.utils.compute_ser(coded_x_int_rep, coded_x_int_zf)
                        coded_ser_lmmse = sn.utils.compute_ser(coded_x_int_rep, coded_x_int_lmmse)
                        coded_ser_dip = sn.utils.compute_ser(coded_x_int_rep, coded_x_int_dip)

                        ### SER Storage
                        ##  Uncoded
                        # sers_zf[i, j] = ser_zf
                        # sers_lmmse[i, j] = ser_lmmse
                        # sers_dip[i, j] = ser_dip
                        ##  Coded
                        coded_sers_zf[m,i,j] = coded_ser_zf
                        coded_sers_lmmse[m,i,j] = coded_ser_lmmse
                        coded_sers_dip[m,i,j] = coded_ser_dip                

                        ### Bit LLR Calculation
                        ##  Uncoded
                        # llr_zf = self.demapper([x_hat_zf, no_eff_zf])
                        # llr_lmmse = self.demapper([x_hat_lmmse, no_eff_lmmse])
                        # llr_dip = self.demapper([x_hat_dip, no_eff_dip])
                        ##  Coded
                        coded_llr_zf = self.demapper([coded_x_hat_zf, coded_no_eff_zf])
                        coded_llr_lmmse = self.demapper([coded_x_hat_lmmse, coded_no_eff_lmmse])
                        coded_llr_dip = self.demapper([coded_x_hat_dip, coded_no_eff_dip])

                        ### LDPC Decoding
                        coded_b_hat_zf = self.LDPC5GDecoder(coded_llr_zf)
                        coded_b_hat_lmmse = self.LDPC5GDecoder(coded_llr_lmmse)
                        coded_b_hat_dip = self.LDPC5GDecoder(coded_llr_dip)

                        ### Hard Decision on Received Bits
                        ##  Uncoded
                        # b_hat_zf = hard_decisions(llr_zf)
                        # b_hat_lmmse = hard_decisions(llr_lmmse)
                        # b_hat_dip = hard_decisions(llr_dip)        

                        ### Bit Errors Calculation
                        ##  Uncoded
                        # bit_err_zf = count_errors(b,b_hat_zf)
                        # bit_err_lmmse = count_errors(b,b_hat_lmmse)
                        # bit_err_dip = count_errors(b,b_hat_dip)
                        ##  Coded
                        coded_bit_err_zf = count_errors(b,coded_b_hat_zf)
                        coded_bit_err_lmmse = count_errors(b,coded_b_hat_lmmse)
                        coded_bit_err_dip = count_errors(b,coded_b_hat_dip)

                        ### BER Calculation
                        ##  Uncoded
                        # ber_zf = compute_ber(b, b_hat_zf)
                        # ber_lmmse = compute_ber(b, b_hat_lmmse)
                        # ber_dip = compute_ber(b, b_hat_dip)
                        ##  Coded
                        coded_ber_zf = compute_ber(b, coded_b_hat_zf)
                        coded_ber_lmmse = compute_ber(b, coded_b_hat_lmmse)
                        coded_ber_dip = compute_ber(b, coded_b_hat_dip)                

                        ### BER Storage
                        ##  Uncoded
                        # bers_zf[i][j] = ber_zf
                        # bers_lmmse[i][j] = ber_lmmse
                        # bers_dip[i][j] = ber_dip
                        ##  Coded
                        coded_bers_zf[m,i,j] = coded_ber_zf
                        coded_bers_lmmse[m,i,j] = coded_ber_lmmse
                        coded_bers_dip[m,i,j] = coded_ber_dip

                        end_time = time.time()
                        time_spent = end_time-start_time
                        # print("{:.3e}|".format(ser_zf).rjust(12), end='')
                        # print("{:.3e}|".format(ber_zf).rjust(12), end='     ')
                        # print("{}|".format(bit_err_zf).rjust(12), end='   ')
                        # print("{:.3e}|".format(ser_lmmse).rjust(12), end='   ')
                        # print("{:.3e}|".format(ber_lmmse).rjust(12), end='        ')
                        # print("{}|".format(bit_err_lmmse).rjust(12), end=' ')
                        # print("{:.3e}|".format(ser_dip).rjust(12), end=' ')
                        # print("{:.3e}|".format(ber_dip).rjust(12), end='      ')
                        # print("{}|".format(bit_err_dip).rjust(12), end='   ')
                        # print("{:.3e}".format(time_spent).rjust(12), end='')
                        # print('|')
                        # print('|', end=' ')
                        # print("CODED|".rjust(12), end='')
                        print("{:.3e}|".format(coded_ser_zf).rjust(12), end='')
                        print("{:.3e}|".format(coded_ber_zf).rjust(12), end='     ')
                        print("{}|".format(coded_bit_err_zf).rjust(12), end='   ')
                        print("{:.3e}|".format(coded_ser_lmmse).rjust(12), end='   ')
                        print("{:.3e}|".format(coded_ber_lmmse).rjust(12), end='        ')
                        print("{}|".format(coded_bit_err_lmmse).rjust(12), end=' ')
                        print("{:.3e}|".format(coded_ser_dip).rjust(12), end=' ')
                        print("{:.3e}|".format(coded_ber_dip).rjust(12), end='      ')
                        print("{}|".format(coded_bit_err_dip).rjust(12), end='   ')
                        print("{:.3e}".format(time_spent).rjust(12), end='')
                        print('|')
                        print('|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|')

                        j = j+1

                        # plt.figure()
                        # plt.axes().set_aspect(1)
                        # plt.grid(True)
                        # plt.title('Flat-Fading Channel Constellation (SNR: {} dB)'.format(EBN0_DB), fontsize=12)
                        # plt.xticks(fontsize=10)
                        # plt.yticks(fontsize=10)
                        # plt.xlabel('REAL', fontsize=10)
                        # plt.ylabel('IMAG', fontsize=10)
                        # plt.scatter(np.real(np.array(x)), np.imag(np.array(x)), s=16, c='b', label='TX')
                        # plt.scatter(np.real(np.array(x_hat_zf)), np.imag(np.array(x_hat_zf)), s=16, c='y', label='ZF')
                        # plt.scatter(np.real(np.array(x_hat_lmmse)), np.imag(np.array(x_hat_lmmse)), s=16, c='g', label='LMMSE')
                        # plt.scatter(np.real(np.array(x_hat_dip)), np.imag(np.array(x_hat_dip)), s=16, c='r', label='DIP')
                        # plt.legend(loc='lower left', fontsize=8)
                        # plt.tight_layout()

                    print('Done')

                    ### Mean SER Calculation
                    ##  Uncoded
                    # sers_zf_mean = np.mean(sers_zf, axis=0)
                    # sers_lmmse_mean = np.mean(sers_lmmse, axis=0)
                    # sers_dip_mean = np.mean(sers_dip, axis=0)
                    ##  Coded
                    coded_sers_zf_mean[m] = tf.math.reduce_mean(coded_sers_zf[m], axis=0)
                    coded_sers_lmmse_mean[m] = tf.math.reduce_mean(coded_sers_lmmse[m], axis=0)
                    coded_sers_dip_mean[m] = tf.math.reduce_mean(coded_sers_dip[m], axis=0)

                    ### Mean BER Calculation
                    ##   Uncoded
                    # bers_zf_mean = np.mean(bers_zf, axis=0)
                    # bers_lmmse_mean = np.mean(bers_lmmse, axis=0)
                    # bers_dip_mean = np.mean(bers_dip, axis=0)
                    ##   Coded
                    coded_bers_zf_mean[m] = tf.math.reduce_mean(coded_bers_zf[m], axis=0)
                    coded_bers_lmmse_mean[m] = tf.math.reduce_mean(coded_bers_lmmse[m], axis=0)
                    coded_bers_dip_mean[m] = tf.math.reduce_mean(coded_bers_dip[m], axis=0)  

                    ##### Plot SER and BER Figures
                    ####  Method 1: Matplot
                    ###   Plot SER
                    ##    Uncoded
                    # plt.figure()
                    # title = "SER: Noncoding MIMO Falt-Fading with ZF, MMSE, DIP Equalizer"
                    # xlabel = "$E_b/N_0$ (dB)"
                    # ylabel = "SER (log)"
                    # plt.title(title, fontsize=12)
                    # plt.xticks(fontsize=10)
                    # plt.yticks(fontsize=10)
                    # plt.xlabel(xlabel, fontsize=10)
                    # plt.ylabel(ylabel, fontsize=10)
                    # plt.grid(which="both")
                    # plt.semilogy(snrs, sers_zf_mean, 'b', label='ZF')
                    # plt.semilogy(snrs, sers_lmmse_mean, 'g', label='LMMSE')
                    # plt.semilogy(snrs, sers_dip_mean, 'r', label='DIP')
                    # plt.legend(loc='lower left', fontsize=8)
                    # plt.tight_layout()

                    ##    Coded
                    plt.figure()
                    title = f"SER: Coding & Corr(Tx{TX_ANT_CORRELATION},Rx{RX_ANT_CORRELATION})"
                    xlabel = "$E_b/N_0$ (dB)"
                    ylabel = "SER (log)"
                    plt.title(title, fontsize=12)
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.xlabel(xlabel, fontsize=10)
                    plt.ylabel(ylabel, fontsize=10)
                    plt.grid(which="both")
                    plt.semilogy(snrs, coded_sers_zf_mean[m], 'b', label='Coded ZF')
                    plt.semilogy(snrs, coded_sers_lmmse_mean[m], 'g', label='Coded LMMSE')
                    plt.semilogy(snrs, coded_sers_dip_mean[m], 'r', label='Coded DIP')
                    plt.legend(loc='lower left', fontsize=8)
                    plt.tight_layout()

                    ##    Combine Uncoded and Coded
                    # plt.figure()
                    # title = "SER: MIMO Falt-Fading with ZF, MMSE, DIP Equalizer"
                    # xlabel = "$E_b/N_0$ (dB)"
                    # ylabel = "SER (log)"
                    # plt.title(title, fontsize=12)
                    # plt.xticks(fontsize=10)
                    # plt.yticks(fontsize=10)
                    # plt.xlabel(xlabel, fontsize=10)
                    # plt.ylabel(ylabel, fontsize=10)
                    # plt.grid(which="both")
                    # plt.semilogy(snrs, sers_zf_mean, 'violet', label='ZF')
                    # plt.semilogy(snrs, sers_lmmse_mean, 'turquoise', label='LMMSE')
                    # plt.semilogy(snrs, sers_dip_mean, 'orange', label='DIP')        
                    # plt.semilogy(snrs, coded_sers_zf_mean, 'blue', label='Coded ZF')
                    # plt.semilogy(snrs, coded_sers_lmmse_mean, 'green', label='Coded LMMSE')
                    # plt.semilogy(snrs, coded_sers_dip_mean, 'red', label='Coded DIP')
                    # plt.legend(loc='lower left', fontsize=8)
                    # plt.tight_layout()

                    ###   Plot BER
                    ##    Uncoded
                    # plt.figure()
                    # title = "BER: Noncoding MIMO Falt-Fading with ZF, MMSE, DIP Equalizer"
                    # xlabel = "$E_b/N_0$ (dB)"
                    # ylabel = "BER (log)"
                    # plt.title(title, fontsize=12)
                    # plt.xticks(fontsize=10)
                    # plt.yticks(fontsize=10)
                    # plt.xlabel(xlabel, fontsize=10)
                    # plt.ylabel(ylabel, fontsize=10)
                    # plt.grid(which="both")
                    # plt.semilogy(snrs, bers_zf_mean, 'b', label='ZF')
                    # plt.semilogy(snrs, bers_lmmse_mean, 'g', label='LMMSE')
                    # plt.semilogy(snrs, bers_dip_mean, 'r', label='DIP')
                    # plt.legend(loc='lower left', fontsize=8)
                    # plt.tight_layout()

                    ##    Coded
                    plt.figure()
                    title = f"BER: Coding & Corr(Tx{TX_ANT_CORRELATION},Rx{RX_ANT_CORRELATION})"
                    xlabel = "$E_b/N_0$ (dB)"
                    ylabel = "BER (log)"
                    plt.title(title, fontsize=12)
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.xlabel(xlabel, fontsize=10)
                    plt.ylabel(ylabel, fontsize=10)
                    plt.grid(which="both")
                    plt.semilogy(snrs, coded_bers_zf_mean[m], 'b', label='Coded ZF')
                    plt.semilogy(snrs, coded_bers_lmmse_mean[m], 'g', label='Coded LMMSE')
                    plt.semilogy(snrs, coded_bers_dip_mean[m], 'r', label='Coded DIP')
                    plt.legend(loc='lower left', fontsize=8)
                    plt.tight_layout()

                    ##    Combine Uncoded and Coded
                    # plt.figure()
                    # title = "BER: MIMO Falt-Fading with ZF, MMSE, DIP Equalizer"
                    # xlabel = "$E_b/N_0$ (dB)"
                    # ylabel = "BER (log)"
                    # plt.title(title, fontsize=12)
                    # plt.xticks(fontsize=10)
                    # plt.yticks(fontsize=10)
                    # plt.xlabel(xlabel, fontsize=10)
                    # plt.ylabel(ylabel, fontsize=10)
                    # plt.grid(which="both")
                    # plt.semilogy(snrs, bers_zf_mean, 'violet', label='ZF')
                    # plt.semilogy(snrs, bers_lmmse_mean, 'turquoise', label='LMMSE')
                    # plt.semilogy(snrs, bers_dip_mean, 'orange', label='DIP')
                    # plt.semilogy(snrs, coded_bers_zf_mean, 'blue', label='Coded ZF')
                    # plt.semilogy(snrs, coded_bers_lmmse_mean, 'green', label='Coded LMMSE')
                    # plt.semilogy(snrs, coded_bers_dip_mean, 'red', label='Coded DIP')
                    # plt.legend(loc='lower left', fontsize=8)
                    # plt.tight_layout()

                    ###   Plot SER and BER, Uncoded and Coded        
                    # plt.figure()
                    # title = f"SER & BER: Coding & Corr(Tx{TX_ANT_CORRELATION},Rx{RX_ANT_CORRELATION})"
                    # xlabel = "$E_b/N_0$ (dB)"
                    # ylabel = "SER/BER (log)"
                    # plt.title(title, fontsize=12)
                    # plt.xticks(fontsize=10)
                    # plt.yticks(fontsize=10)
                    # plt.xlabel(xlabel, fontsize=10)
                    # plt.ylabel(ylabel, fontsize=10)
                    # plt.grid(which="both")
                    # plt.semilogy(snrs, sers_zf_mean, 'violet', label='ZF SER')
                    # plt.semilogy(snrs, sers_lmmse_mean, 'turquoise', label='LMMSE SER')
                    # plt.semilogy(snrs, sers_dip_mean, 'orange', label='DIP SER')        
                    # plt.semilogy(snrs, coded_sers_zf_mean[m], 'dodgerblue', label='Coded ZF SER')
                    # plt.semilogy(snrs, coded_sers_lmmse_mean[m], 'lime', label='Coded LMMSE SER')
                    # plt.semilogy(snrs, coded_sers_dip_mean[m], 'darkred', label='Coded DIP SER')
                    # plt.semilogy(snrs, bers_zf_mean, 'purple', label='ZF BER')
                    # plt.semilogy(snrs, bers_lmmse_mean, 'lightseagreen', label='LMMSE BER')
                    # plt.semilogy(snrs, bers_dip_mean, 'gold', label='DIP BER')
                    # plt.semilogy(snrs, coded_bers_zf_mean[m], 'blue', label='Coded ZF BER')
                    # plt.semilogy(snrs, coded_bers_lmmse_mean[m], 'green', label='Coded LMMSE BER')
                    # plt.semilogy(snrs, coded_bers_dip_mean[m], 'red', label='Coded DIP BER')
                    # plt.legend(loc='lower left', fontsize=8)
                    # plt.tight_layout()

                    # plt.show()

                    ##  Method 2: Bokeh
                    #   Plot SER and BER, Uncoded and Coded
                    # plotycodeycorr(x=snrs, 
                                    # y1=sers_zf_mean,
                                    # y2=sers_lmmse_mean,
                                    # y3=sers_dip_mean,
                                    # y4=coded_sers_zf_mean[m],
                                    # y5=coded_sers_lmmse_mean[m],
                                    # y6=coded_sers_dip_mean[m],
                                    # y7=bers_zf_mean,
                                    # y8=bers_lmmse_mean,
                                    # y9=bers_dip_mean,
                                    # y10=coded_bers_zf_mean[m],
                                    # y11=coded_bers_lmmse_mean[m],
                                    # y12=coded_bers_dip_mean[m],
                                    # y_label="SER/BER (log)",
                                    # title="SER & BER: Coding & Corr({TX_ANT_CORRELATION},{RX_ANT_CORRELATION})",
                                    # filename="SER&BER-NCodeYCorr({TX_ANT_CORRELATION},{RX_ANT_CORRELATION}).html")

                ### Spatial Correlation Loop Variable m
                m = m+1

        ### Plot SER and BER Seperately at different Spatial Correlation
        #  SER
        # plt.figure()
        # title = "SER: Coding & Corr"
        # xlabel = "$E_b/N_0$ (dB)"
        # ylabel = "SER (log)"
        # plt.title(title, fontsize=12)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel(xlabel, fontsize=10)
        # plt.ylabel(ylabel, fontsize=10)
        # plt.grid(which="both")
        # for m in range(NUM_COR_GROUP):
        #     plt.semilogy(snrs, coded_sers_zf_mean[m], label='ZF SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     plt.semilogy(snrs, coded_sers_lmmse_mean[m], label='LMMSE SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     plt.semilogy(snrs, coded_sers_dip_mean[m], label='DIP SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))     
        #     plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
        #     plt.tight_layout()
        # plt.show()
        
        ##  BER 
        # plt.figure()
        # title = "BER: Coding & Corr"
        # xlabel = "$E_b/N_0$ (dB)"
        # ylabel = "SER/BER (log)"
        # plt.title(title, fontsize=12)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel(xlabel, fontsize=10)
        # plt.ylabel(ylabel, fontsize=10)
        # plt.grid(which="both")
        # for m in range(NUM_COR_GROUP):
        #     plt.semilogy(snrs, coded_bers_zf_mean[m], label='ZF BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     plt.semilogy(snrs, coded_bers_lmmse_mean[m], label='LMMSE BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     plt.semilogy(snrs, coded_bers_dip_mean[m], label='DIP BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
        #     plt.tight_layout()
        # plt.show()
        
        ##  SER & BER
        # plt.figure()
        # title = "SER & BER: Coding & Corr"
        # xlabel = "$E_b/N_0$ (dB)"
        # ylabel = "SER/BER (log)"
        # plt.title(title, fontsize=12)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel(xlabel, fontsize=10)
        # plt.ylabel(ylabel, fontsize=10)
        # plt.grid(which="both")
        # for m in range(NUM_COR_GROUP):
        #     plt.semilogy(snrs, coded_bers_zf_mean[m], label='ZF BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     plt.semilogy(snrs, coded_bers_lmmse_mean[m], label='LMMSE BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     plt.semilogy(snrs, coded_bers_dip_mean[m], label='DIP BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     # plt.semilogy(snrs, coded_sers_zf_mean[m], label='ZF SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     # plt.semilogy(snrs, coded_sers_lmmse_mean[m], label='LMMSE SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
        #     # plt.semilogy(snrs, coded_sers_dip_mean[m], label='DIP SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1])) 
        #     plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
        #     plt.tight_layout()
        # plt.show()

        return snrs, coded_sers_zf_mean, coded_sers_lmmse_mean, coded_sers_dip_mean

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
from NCodeNCorrPlot import ncodencorrplot

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

class ncodencorr(Model): # Inherits from Keras Model
    def __init__(self,
                 Block_Length,
                 NUM_BITS_PER_SYMBOL,
                 CONSTELLATION_TYPE,
                 DEMAPPING_METHOD,
                 NUM_RX_ANT,
                 NUM_TX_ANT):

        super().__init__() # Must call the Keras model initializer
        
        self.Block_Length = Block_Length
        self.NUM_BITS_PER_SYMBOL = NUM_BITS_PER_SYMBOL
        self.CONSTELLATION_TYPE = CONSTELLATION_TYPE
        self.DEMAPPING_METHOD = DEMAPPING_METHOD
        self.NUM_RX_ANT = NUM_RX_ANT
        self.NUM_TX_ANT = NUM_TX_ANT
        self.constellation = sn.mapping.Constellation(constellation_type=self.CONSTELLATION_TYPE,
                                                      num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL)
        self.mapper = sn.mapping.Mapper(constellation_type=self.CONSTELLATION_TYPE,
                                        num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
                                        return_indices=True)
        self.demapper = sn.mapping.Demapper(demapping_method=self.DEMAPPING_METHOD,
                                            constellation_type=self.CONSTELLATION_TYPE,
                                            num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL)
        self.SymbolDemapper = sn.mapping.SymbolDemapper(constellation_type=self.CONSTELLATION_TYPE,
                                                        num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
                                                        hard_out=True)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        self.flatfading_channel = sn.channel.FlatFadingChannel(num_tx_ant=self.NUM_TX_ANT,
                                                               num_rx_ant=self.NUM_RX_ANT,
                                                               add_awgn=True,
                                                               return_channel=True)
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
        
        snrs = np.linspace(EBN0_DB_MIN,EBN0_DB_MAX,NUM_EBN0_POINTS)
        sers_zf = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        sers_lmmse = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        sers_dip = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        bers_zf = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        bers_lmmse = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))
        bers_dip = np.empty((NUM_DATA_GROUP, NUM_EBN0_POINTS))

        for i in range(0, NUM_DATA_GROUP):
            print('|                                                                            Data Group {}                                                                           |'.format(i+1))
            print('|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|')
            print('|  EBN0[DB]  |  SER(ZF)  |  BER(ZF)  | Bit Errors(ZF) |  SER(LMMSE)  |  BER(LMMSE)  | Bit Errors(LMMSE) |  SER(DIP)  |  BER(DIP)  | Bit Errors(DIP) | Time Spent(s) |')
            print('|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|')
            b = self.binary_source([BATCH_SIZE,self.NUM_TX_ANT,self.Block_Length])
            x, x_int_rep =  self.mapper(b)

            shape = tf.shape(x)
            x_reshape = tf.reshape(x,[-1, self.NUM_TX_ANT])
            
            j = 0

            for EBN0_DB in np.linspace(EBN0_DB_MIN,EBN0_DB_MAX,NUM_EBN0_POINTS):
                start_time = time.time()
                print('|', end=' ')
                print("{}|".format(EBN0_DB).rjust(12), end='')
                
                # No Channel Coding Used: Coderate=1.0
                no = sn.utils.ebnodb2no(ebno_db=EBN0_DB,
                                        num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
                                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here

                # y and h are the Channel Output and Channel Realizations, respectively
                y, h = self.flatfading_channel([x_reshape, no]) # type: ignore
                s = tf.cast(no*tf.eye(self.NUM_RX_ANT, self.NUM_RX_ANT), y.dtype)

                ### Deep Image Prior Eqaulizer
                ##  Complex to Real
                y_shape = tf.shape(y)
                y_reshape = tf.reshape(y,[-1, self.NUM_RX_ANT])
                y_reshape_real_part = tf.math.real(y_reshape)
                y_reshape_image_part = tf.math.imag(y_reshape)
                y_reshape_real = tf.concat([y_reshape_real_part,y_reshape_image_part], axis=1)
                h_shape = tf.shape(h)
                h_reshape = tf.reshape(h,[-1, self.NUM_RX_ANT, self.NUM_TX_ANT])
                h_reshape_real_part = tf.math.real(h_reshape)
                h_reshape_imag_part = tf.math.imag(h_reshape)
                h_reshape_real = tf.concat([tf.concat([h_reshape_real_part, tf.multiply(h_reshape_imag_part,-1)], axis=2),
                                            tf.concat([h_reshape_imag_part, h_reshape_real_part], axis=2)],axis=1)
                ##  DIP Equalizer
                x_dip_ay,num_stop_point = self.dip.DIP(np.array(y_reshape_real),np.array(h_reshape_real))
                x_dip_ay_real_part,x_dip_ay_imag_part = np.split(x_dip_ay, indices_or_sections=2, axis=1)
                x_hat_dip = tf.cast(tf.reshape(tf.complex(x_dip_ay_real_part,x_dip_ay_imag_part), shape), dtype=tf.complex64)
                ##  Assume noise variance equal to channel noise
                no_eff_dip = no*tf.ones(shape)

                ### Zero-Forcing Equalizer
                x_hat_zf, no_eff_zf = zf_equalizer(y, h, s)
                no_eff_zf = tf.reshape(no_eff_zf, shape)
                x_hat_zf = tf.reshape(x_hat_zf, shape)

                ### LMMSE Equalizer
                x_hat_lmmse, no_eff_lmmse = lmmse_equalizer(y, h, s)
                no_eff_lmmse = tf.reshape(no_eff_lmmse, shape)
                x_hat_lmmse = tf.reshape(x_hat_lmmse, shape)

                ### Soft Decision Outputs Received Symbols(Integer)
                x_int_zf = self.SymbolDemapper([x_hat_zf, no_eff_zf])
                x_int_lmmse = self.SymbolDemapper([x_hat_lmmse, no_eff_lmmse])
                x_int_dip = self.SymbolDemapper([x_hat_dip, no_eff_dip])

                ### SER Calculation
                ser_zf = sn.utils.compute_ser(x_int_rep, x_int_zf)             
                ser_lmmse = sn.utils.compute_ser(x_int_rep, x_int_lmmse)    
                ser_dip = sn.utils.compute_ser(x_int_rep, x_int_dip)   

                ### SER Storage
                sers_zf[i, j] = ser_zf
                sers_lmmse[i, j] = ser_lmmse
                sers_dip[i, j] = ser_dip

                # Bit LLR Calculation
                llr_zf = self.demapper([x_hat_zf, no_eff_zf])
                llr_lmmse = self.demapper([x_hat_lmmse, no_eff_lmmse])
                llr_dip = self.demapper([x_hat_dip, no_eff_dip])

                ### Hard Decision on Received Bits
                b_hat_zf = hard_decisions(llr_zf)
                b_hat_lmmse = hard_decisions(llr_lmmse)
                b_hat_dip = hard_decisions(llr_dip)

                ### Bit Errors Calculation
                bit_err_zf = count_errors(b,b_hat_zf)
                bit_err_lmmse = count_errors(b,b_hat_lmmse)
                bit_err_dip = count_errors(b,b_hat_dip)

                ### BER Calculation
                ber_zf = compute_ber(b, b_hat_zf)
                ber_lmmse = compute_ber(b, b_hat_lmmse)
                ber_dip = compute_ber(b, b_hat_dip)

                ### BER Storage
                bers_zf[i][j] = ber_zf
                bers_lmmse[i][j] = ber_lmmse
                bers_dip[i][j] = ber_dip

                end_time = time.time()
                time_spent = end_time-start_time
                print("{:.3e}|".format(ser_zf).rjust(12), end='')
                print("{:.3e}|".format(ber_zf).rjust(12), end='     ')
                print("{}|".format(bit_err_zf).rjust(12), end='   ')
                print("{:.3e}|".format(ser_lmmse).rjust(12), end='   ')
                print("{:.3e}|".format(ber_lmmse).rjust(12), end='        ')
                print("{}|".format(bit_err_lmmse).rjust(12), end=' ')
                print("{:.3e}|".format(ser_dip).rjust(12), end=' ')
                print("{:.3e}|".format(ber_dip).rjust(12), end='      ')
                print("{}|".format(bit_err_dip).rjust(12), end='   ')
                print("{:.3e}".format(time_spent).rjust(12), end='')
                print('|')
                print('|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|')

                j = j+1

                plt.figure()
                plt.axes().set_aspect(1)
                plt.grid(True)
                plt.title('Flat-Fading Channel Constellation (SNR: {} dB)'.format(EBN0_DB), fontsize=12)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.xlabel('REAL', fontsize=10)
                plt.ylabel('IMAG', fontsize=10)
                plt.scatter(np.real(np.array(x)), np.imag(np.array(x)), s=16, c='b', label='TX')
                plt.scatter(np.real(np.array(x_hat_zf)), np.imag(np.array(x_hat_zf)), s=16, c='y', label='ZF')
                plt.scatter(np.real(np.array(x_hat_lmmse)), np.imag(np.array(x_hat_lmmse)), s=16, c='g', label='LMMSE')
                plt.scatter(np.real(np.array(x_hat_dip)), np.imag(np.array(x_hat_dip)), s=16, c='r', label='DIP')
                plt.legend(loc='lower left', fontsize=8)
                plt.tight_layout()

            print('Done')

        ### Mean SER Calculation
        sers_zf_mean = np.mean(sers_zf, axis=0)
        sers_lmmse_mean = np.mean(sers_lmmse, axis=0)
        sers_dip_mean = np.mean(sers_dip, axis=0)

        ### Mean BER Calculation
        bers_zf_mean = np.mean(bers_zf, axis=0)
        bers_lmmse_mean = np.mean(bers_lmmse, axis=0)
        bers_dip_mean = np.mean(bers_dip, axis=0)

        ### Plot SER and BER Figures
        ##  Method 1: Matplot
        #   Plot SER
        plt.figure()
        title = "SER: No Coding & No Sptatial Correlation"
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

        #   Plot BER
        plt.figure()
        title = "BER: No Coding & No Sptatial Correlation"
        xlabel = "$E_b/N_0$ (dB)"
        ylabel = "BER (log)"
        plt.title(title, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(which="both")
        plt.semilogy(snrs, bers_zf_mean, 'b', label='ZF')
        plt.semilogy(snrs, bers_lmmse_mean, 'g', label='LMMSE')
        plt.semilogy(snrs, bers_dip_mean, 'r', label='DIP')
        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()

        plt.show()

        ##  Method 2: Bokeh
        #   Plot SER
        # ncodencorrplot(x=snrs, 
        #             y1=sers_zf_mean,
        #             y2=sers_lmmse_mean,
        #             y3=sers_dip_mean,
        #             y_label="SER (log)",
        #             title="SER: No Coding & No Sptatial Correlation",
        #             filename="SER-NCodeNCorr.html")
        # #   Plot BER
        # ncodencorrplot(x=snrs, 
        #             y1=bers_zf_mean,
        #             y2=bers_lmmse_mean,
        #             y3=bers_dip_mean,
        #             y_label="BER (log)",
        #             title="BER: No Coding & No Sptatial Correlation",
        #             filename="BER-NCodeNCorr.html")

        return snrs, sers_zf_mean, sers_lmmse_mean, sers_dip_mean
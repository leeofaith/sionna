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
from NCodeYCorrPlot import ncodeycorrplot

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
from sionna.channel.utils import exp_corr_mat
from sionna.channel import KroneckerModel

class ncodencorr(Model): # Inherits from Keras Model
    def __init__(self,
                 Block_Length,
                 NUM_BITS_PER_SYMBOL,
                 CONSTELLATION_TYPE,
                 DEMAPPING_METHOD,
                 NUM_RX_ANT,
                 NUM_TX_ANT,
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
                                  LR=0.01,
                                  buffer_size=30,
                                  threshold=0.001,
                                  stop=True)

    def __call__(self,
                 NUM_DATA_GROUP,
                 BATCH_SIZE,
                 EBN0_DB_MIN,
                 EBN0_DB_MAX,
                 NUM_EBN0_POINTS):

        ### Number of Spatial Correlation Group
        NUM_COR_GROUP = (self.CORRELATION_INDEX_POINT)*(self.CORRELATION_INDEX_POINT)
        COR_GROUP =  ([0.1,0.1],[0.1,0.5],[0.1,0.9],[0.5,0.1],[0.5,0.5],[0.5,0.9],[0.9,0.1],[0.9,0.5],[0.9,0.9])

        snrs = np.linspace(EBN0_DB_MIN,EBN0_DB_MAX,NUM_EBN0_POINTS)
        ### Uncode Variables Definition
        sers_zf = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        sers_lmmse = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        sers_dip = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        bers_zf = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        bers_lmmse = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        bers_dip = np.empty((NUM_COR_GROUP, NUM_DATA_GROUP, NUM_EBN0_POINTS))
        sers_zf_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        sers_lmmse_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        sers_dip_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        bers_zf_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        bers_lmmse_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))
        bers_dip_mean = np.empty((NUM_COR_GROUP, NUM_EBN0_POINTS))

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
                        ### Channel Transmission
                        ##  Uncoded
                        y, h = self.flatfading_channel([x_reshape, no]) # type: ignore
                        s = tf.cast(no*tf.eye(self.NUM_RX_ANT, self.NUM_RX_ANT), y.dtype)

                        ### Deep Image Prior Eqaulizer
                        ##  Complex to Real
                        y_shape = tf.shape(y)
                        y_reshape = tf.reshape(y,[-1, self.NUM_RX_ANT])
                        y_reshape_real_part = np.real(y_reshape)
                        y_reshape_image_part = np.imag(y_reshape)
                        y_reshape_real = np.concatenate([y_reshape_real_part,y_reshape_image_part], axis=1)
                        h_shape = tf.shape(h)
                        h_reshape = tf.reshape(h,[-1, self.NUM_RX_ANT, self.NUM_TX_ANT])
                        h_reshape_real_part = np.real(h_reshape)
                        h_reshape_imag_part = np.imag(h_reshape)
                        h_reshape_real = np.concatenate([np.concatenate([h_reshape_real_part, -h_reshape_imag_part], axis=2), np.concatenate([h_reshape_imag_part, h_reshape_real_part], axis=2)],axis=1)
                        ##  DIP Equalizer
                        x_dip_ay,num_stop_point = self.dip.DIP(y_reshape_real,h_reshape_real)
                        x_dip_ay_real_part,x_dip_ay_imag_part = np.split(x_dip_ay, indices_or_sections=2, axis=1)
                        x_hat_dip = tf.cast(tf.reshape(tf.complex(x_dip_ay_real_part,x_dip_ay_imag_part), shape), dtype=tf.complex64)
                        ##  Assume noise variance equal to channel noise
                        no_eff_dip = no*np.ones(shape)

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
                        sers_zf[m, i, j] = ser_zf
                        sers_lmmse[m, i, j] = ser_lmmse
                        sers_dip[m, i, j] = ser_dip            

                        ### Bit LLR Calculation
                        llr_zf = self.demapper([x_hat_zf, no_eff_zf])
                        llr_lmmse = self.demapper([x_hat_lmmse, no_eff_lmmse])
                        llr_dip = self.demapper([tf.cast(x_hat_dip, dtype=tf.complex64), no_eff_dip])

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
                        bers_zf[m][i][j] = ber_zf
                        bers_lmmse[m][i][j] = ber_lmmse
                        bers_dip[m][i][j] = ber_dip

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

                        ### EBN0_DB Loop Variable
                        j = j+1

                        plt.figure()
                        plt.axes().set_aspect(1)
                        plt.grid(True)
                        plt.title('Flat-Fading Channel Constellation (SNR: {} dB)'.format(EBN0_DB), fontsize=12)
                        plt.xticks(fontsize=10)
                        plt.yticks(fontsize=10)
                        plt.xlabel('REAL', fontsize=10)
                        plt.ylabel('IMAG', fontsize=10)
                        plt.scatter(np.real(x), np.imag(x), s=16, c='b', label='TX')
                        plt.scatter(np.real(x_hat_zf), np.imag(x_hat_zf), s=16, c='y', label='ZF')
                        plt.scatter(np.real(x_hat_lmmse), np.imag(x_hat_lmmse), s=16, c='g', label='LMMSE')
                        plt.scatter(np.real(x_hat_dip), np.imag(x_hat_dip), s=16, c='r', label='DIP')
                        plt.legend(loc='lower left', fontsize=8)
                        plt.tight_layout()

                    print('Done')

                    ### Mean SER Calculation
                    ##  sers_xxx(m, i, j): (spatial, data group, SER at different SNR), means at same spatial correlation (m) but different data group (i)
                    #   Uncoded
                    sers_zf_mean[m] = np.mean(sers_zf[m], axis=0)
                    sers_lmmse_mean[m] = np.mean(sers_lmmse[m], axis=0)
                    sers_dip_mean[m] = np.mean(sers_dip[m], axis=0)     

                    ### Mean BER Calculation
                    ##  bers_xxx(m, i, j): (spatial, data group, BER at different SNR), means at same spatial correlation (m) but different data group (i)
                    #   Uncoded
                    bers_zf_mean[m] = np.mean(bers_zf[m], axis=0)
                    bers_lmmse_mean[m] = np.mean(bers_lmmse[m], axis=0)
                    bers_dip_mean[m] = np.mean(bers_dip[m], axis=0)     

                    ##### Plot SER and BER Figures
                    ####  Method 1: Matplot
                    ###   Plot SER
                    ##    Uncoded
                    plt.figure()
                    title = f"SER: No Coding & Corr(Tx{TX_ANT_CORRELATION},Rx{RX_ANT_CORRELATION})"
                    xlabel = "$E_b/N_0$ (dB)"
                    ylabel = "SER (log)"
                    plt.title(title, fontsize=12)
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.xlabel(xlabel, fontsize=10)
                    plt.ylabel(ylabel, fontsize=10)
                    plt.grid(which="both")
                    plt.semilogy(snrs, sers_zf_mean[m], 'b', label='ZF')
                    plt.semilogy(snrs, sers_lmmse_mean[m], 'g', label='LMMSE')
                    plt.semilogy(snrs, sers_dip_mean[m], 'r', label='DIP')
                    plt.legend(loc='lower left', fontsize=8)
                    plt.tight_layout()

                    ###   Plot BER
                    ##    Uncoded
                    plt.figure()
                    title = f"BER: No Coding & Corr({TX_ANT_CORRELATION},{RX_ANT_CORRELATION})"
                    xlabel = "$E_b/N_0$ (dB)"
                    ylabel = "BER (log)"
                    plt.title(title, fontsize=12)
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.xlabel(xlabel, fontsize=10)
                    plt.ylabel(ylabel, fontsize=10)
                    plt.grid(which="both")
                    plt.semilogy(snrs, bers_zf_mean[m], 'b', label='ZF')
                    plt.semilogy(snrs, bers_lmmse_mean[m], 'g', label='LMMSE')
                    plt.semilogy(snrs, bers_dip_mean[m], 'r', label='DIP')
                    plt.legend(loc='lower left', fontsize=8)
                    plt.tight_layout()

                    ###   Plot SER and BER, Uncoded and Coded        
                    plt.figure()
                    title = f"SER & BER: No Coding & Corr({TX_ANT_CORRELATION},{RX_ANT_CORRELATION})"
                    xlabel = "$E_b/N_0$ (dB)"
                    ylabel = "SER/BER (log)"
                    plt.title(title, fontsize=12)
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.xlabel(xlabel, fontsize=10)
                    plt.ylabel(ylabel, fontsize=10)
                    plt.grid(which="both")
                    plt.semilogy(snrs, sers_zf_mean[m], 'violet', label='ZF SER')
                    plt.semilogy(snrs, sers_lmmse_mean[m], 'turquoise', label='LMMSE SER')
                    plt.semilogy(snrs, sers_dip_mean[m], 'orange', label='DIP SER')        
                    plt.semilogy(snrs, bers_zf_mean[m], 'purple', label='ZF BER')
                    plt.semilogy(snrs, bers_lmmse_mean[m], 'lightseagreen', label='LMMSE BER')
                    plt.semilogy(snrs, bers_dip_mean[m], 'gold', label='DIP BER')
                    plt.legend(loc='lower left', fontsize=8)
                    plt.tight_layout()

                    plt.show()

                    ##  Method 2: Bokeh
                    #   Plot SER and BER, Uncoded and Coded
                    ncodeycorrplot(x=snrs,
                                    y1=sers_zf_mean[m],
                                    y2=sers_lmmse_mean[m],
                                    y3=sers_dip_mean[m],
                                    y4=bers_zf_mean[m],
                                    y5=bers_lmmse_mean[m],
                                    y6=bers_dip_mean[m],
                                    y_label="SER/BER (log)",
                                    title=f"SER & BER: No Coding & Corr({TX_ANT_CORRELATION},{RX_ANT_CORRELATION})",
                                    filename=f"SER&BER-NCodeYCorr({TX_ANT_CORRELATION},{RX_ANT_CORRELATION}).html")
                    
                ### Spatial Correlation Loop Variable m
                m = m+1
        
        ### Plot SER and BER Seperately at different Spatial Correlation
        ##  SER
        plt.figure()
        title = "SER: No Coding & Corr"
        xlabel = "$E_b/N_0$ (dB)"
        ylabel = "SER (log)"
        plt.title(title, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(which="both")
        for m in range(NUM_COR_GROUP):
            plt.semilogy(snrs, sers_zf_mean[m], label='ZF SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, sers_lmmse_mean[m], label='LMMSE SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, sers_dip_mean[m], label='DIP SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))     
            plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
            plt.tight_layout()
        plt.show()
        
        ##  BER 
        plt.figure()
        title = "BER: No Coding & Corr"
        xlabel = "$E_b/N_0$ (dB)"
        ylabel = "SER/BER (log)"
        plt.title(title, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(which="both")
        for m in range(NUM_COR_GROUP):
            plt.semilogy(snrs, bers_zf_mean[m], label='ZF BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, bers_lmmse_mean[m], label='LMMSE BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, bers_dip_mean[m], label='DIP BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
            plt.tight_layout()
        plt.show()
        
        ##  SER & BER
        plt.figure()
        title = "SER & BER: No Coding & Corr"
        xlabel = "$E_b/N_0$ (dB)"
        ylabel = "SER/BER (log)"
        plt.title(title, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(which="both")
        for m in range(NUM_COR_GROUP):
            plt.semilogy(snrs, bers_zf_mean[m], label='ZF BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, bers_lmmse_mean[m], label='LMMSE BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, bers_dip_mean[m], label='DIP BER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, sers_zf_mean[m], label='ZF SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, sers_lmmse_mean[m], label='LMMSE SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1]))
            plt.semilogy(snrs, sers_dip_mean[m], label='DIP SER ({},{})'.format(COR_GROUP[m][0],COR_GROUP[m][1])) 
            plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
            plt.tight_layout()
        plt.show()

        return snrs, sers_zf_mean, sers_lmmse_mean, sers_dip_mean

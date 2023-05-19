import numpy as np
import torch
from torch import nn



class DeepImagePrior(object):
    def __init__(self,num_rx_ant,num_tx_ant,M, iteration,LR,buffer_size,threshold,stop):
        
        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant    ####number of transmitted symbol in real domain;
        self.M = M                  ####modulation order, 4 for 4qam, 16 for 16qam
        self.iteration = iteration  ####number of max iterations used for DIP
        self.LR = LR                ####Learning rate,  typically set to 0.01; Control step size of updating the model parameters at each iteration
        self.buffer_size = buffer_size    ###iterations stored,  typically set to 30
        self.threshold = threshold        ###Threshold of DIP stop,, typically set to 0.001
        self.stop = stop                  ###True
        constellation = np.linspace(int(-np.sqrt(M) + 1), int(np.sqrt(M) - 1), int(np.sqrt(M)))
        alpha = np.sqrt((constellation ** 2).mean())
        constellation /= (alpha * np.sqrt(2))
        self.constellation = constellation
        constellation_expanded = np.expand_dims(self.constellation, axis=1)
        constellation_expanded= np.repeat(constellation_expanded[None,...],1,axis=0)

        constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), self.num_tx_ant, axis=1)

        self.constellation_expanded =torch.from_numpy(constellation_expanded)
        self.constellation_expanded_transpose = torch.from_numpy(constellation_expanded_transpose)

    def QAM_const(self):
        mod_n = self.M
        sqrt_mod_n = np.int(np.sqrt(mod_n))
        real_qam_consts = np.empty((mod_n), dtype=np.int64)
        imag_qam_consts = np.empty((mod_n), dtype=np.int64)
        for i in range(sqrt_mod_n):
            for j in range(sqrt_mod_n):
                    index = sqrt_mod_n*i + j
                    real_qam_consts[index] = i
                    imag_qam_consts[index] = j

        return(self.constellation[real_qam_consts], self.constellation[imag_qam_consts])
    
    def ser(self,x_hat, x_true):
        
        real_QAM_const,imag_QAM_const = self.QAM_const()
        x_real, x_imag = np.split(x_hat, 2, -1)
        x_real = np.expand_dims(x_real,-1).repeat(real_QAM_const.size,-1)
        x_imag = np.expand_dims(x_imag,-1).repeat(imag_QAM_const.size,-1)
    
        x_real = np.power(x_real - real_QAM_const, 2)
        x_imag = np.power(x_imag - imag_QAM_const, 2)
        x_dist = x_real + x_imag
        estim_indices = np.argmin(x_dist, axis=-1)
        
        
        x_real_true, x_imag_true = np.split(x_true, 2, -1)
        x_real_true = np.expand_dims(x_real_true,-1).repeat(real_QAM_const.size,-1)
        x_imag_true = np.expand_dims(x_imag_true,-1).repeat(imag_QAM_const.size,-1)
    
        x_real_true = np.power(x_real_true - real_QAM_const, 2)
        x_imag_true = np.power(x_imag_true - imag_QAM_const, 2)
        x_dist_true = x_real_true + x_imag_true
        true_indices = np.argmin(x_dist_true, axis=-1)
        
         # estim_indices = joint_indices(x_hat_indices,constellation)
        ser = np.sum(true_indices!=estim_indices)/true_indices.size
        return ser


    def DIP(self,Y,H):
        
        torch.backends.cudnn.enabled = True # Enable CUDA DNN library (cuDNN) to accelerate NN operations
        torch.backends.cudnn.benchmark = True # Find the most suitable cuDNN algorithm to maximize performance
        dtype = torch.FloatTensor
        batch_size = H.shape[0]
        x_dip_ay = np.empty((batch_size,self.num_tx_ant,2,1))
        num_stop_point = []
        
        for bs in range(batch_size):

            i = 0            
            flag = False
            
            net = Decoder(self.num_tx_ant,self.num_tx_ant).type(dtype)    ### Define the Neural network
            mse = torch.nn.MSELoss().type(dtype)        ### Loss function: Mean Squared Error; MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
            # Instance Adam Optimizer
            optimizer = torch.optim.Adam(net.parameters(), lr= self.LR)     ###Adam optimizer
            # Reset the Weight of the convolutional or linear layers to random values
            net.apply(weight_reset)                     ### Reset the neural network parameters
            # 正态分布数组size(1,4),可以改成均匀分布等其他分布
            net_input = torch.randn(1,4)    ### Random input for DIP
            # print('Y shape=',Y.shape)
            # print('Y =',Y)
            # print('H shape=',H.shape)
            # print('H =',H)
            # y_torch = torch.from_numpy((Y[bs]).numpy()).reshape(1,1,2*self.num_rx_ant,1).type(dtype)
            # H_torch = torch.from_numpy((H[bs]).numpy()).reshape(1,1,2*self.num_rx_ant,2*self.num_tx_ant).type(dtype)

            y_torch = torch.from_numpy((Y[bs]).numpy()).type(dtype)
            H_torch = torch.from_numpy((H[bs]).numpy()).type(dtype)
            # print('y_torch shape=',y_torch.shape)
            # print('y_torch =',y_torch)
            # print('H_torch shape=',H_torch.shape)
            # print('H_torch =',H_torch)
            
            variance_history = []
            earlystop = EarlyStop(size= self.buffer_size)
            
            while i < self.iteration and flag==False :
                
                i+=1 
                
                # step 1, update network:
                # Zeroize Gradient
                optimizer.zero_grad()
                net_output = net(net_input).type(dtype)
                # print('net_output shape =', net_output.shape)
                # print('net_output =', net_output)
                max_constellation= np.max(self.constellation)
                # print('max_constellation shape =', max_constellation.shape)
                # print('max_constellation =', max_constellation)
                out = net_output*max_constellation
                # print('out =', out)
                # out = net(net_input).type(dtype)*np.max(self.constellation)
                # print('out1 shape =',out1.shape)
                # print('out1 =',out1)
                out1 = torch.reshape(out.repeat(self.num_rx_ant,1,1,1),[self.num_rx_ant,self.num_tx_ant,2,1])
                # print('out1 shape =',out1.shape)
                # print('out1 =',out1)
                result = torch.matmul(H_torch,out1)
                # print('result shape =',result.shape)
                # print('result =',result)
                Y_hat = torch.sum(result, axis=1)
                # print('Y_hat shape =',Y_hat.shape)
                # print('Y_hat =',Y_hat)
                # Y_hat = torch.matmul(H_torch,out1).type(dtype)

                # print('Y_hat =',Y_hat)
                # print('y_torch =',y_torch)
                total_loss = mse(Y_hat,y_torch)
                total_loss.backward()
                optimizer.step()
                # print('total_loss =', total_loss)
                # print('Y_hat =',Y_hat)
                # print('out =', out)
                x_dip = out.detach().cpu().numpy()
                # print('x_dip shape =', x_dip.shape)
                # print('x_dip =', x_dip)
                if  self.stop is True:
                    r_img_np = x_dip.reshape(-1)
                    # print('r_img_np =', r_img_np)
                    earlystop.update_img_collection(r_img_np)
                    img_collection = earlystop.get_img_collection()
                    # print('img_collection[0] =', img_collection[0])
                    # print('img_collection =', img_collection)
                    if len(img_collection) ==  self.buffer_size:
                        ave_img = np.mean(img_collection, axis=0) # Average buffer_sized img_colletction
                        # print('ave_img =', ave_img)
                        # print('ave_img length =', len(ave_img))
                        variance = []
                        for tmp in img_collection:
                            # print('tmp =', tmp)
                            # print('tmp length =', len(tmp))
                            variance.append(earlystop.Var_cal(ave_img,tmp))
                        cur_var = np.mean(variance)
                        variance_history.append(cur_var)
                    else:
                        cur_var = 0
                    
                    if cur_var != 0 and cur_var <  self.threshold:
                        num_stop_point.append(i)
                        flag = True

            # H_torch_qualified = H_torch.reshape(self.num_rx_ant,self.num_tx_ant,2,2)
            # out1 = out1.reshape(self.num_rx_ant,self.num_tx_ant,2,2)
            # print('out1[1] shape=',out1[1].shape)
            # print('out1[1] =',out1[1])
            # print('H_torch shape=',H_torch.shape)
            # print('H_torch =',H_torch)
            # print('total_loss =',total_loss)
            # print('Y_hat shape=',Y_hat.shape)
            # print('Y_hat =',Y_hat)
            # print('y_torch =',y_torch)            
            # print('x_dip =', x_dip)            
            x_dip_ay[bs] = out1[1].detach().numpy()
            # print('x_dip_ay[{}] ={}'.format(bs, x_dip_ay[bs]))
        return x_dip_ay,num_stop_point

class Decoder(nn.Module):
    def __init__(self,num_rx_ant,num_tx_ant):
        super().__init__()
        
        self.nn1 = nn.Linear(4,8) # Define numbers of input (4) and output (8) of fully connected layer
                                  # Weight.shape = (number of output (8),number of input (4))
                                  # Bias.shape = number of output (8)
        self.nn2 = nn.Linear(8,16)
        self.nn3 = nn.Linear(16,32)
        self.nn4 = nn.Linear(32,(2*num_tx_ant))
        self.act = nn.Tanh() # Define Activation Function: Tanh()
         
    def forward(self,x): 
        o1 = self.act(self.nn1(x)) 
        o2 = self.act(self.nn2(o1)) 
        o3 = self.act(self.nn3(o2)) 
        o4 = self.act(self.nn4(o3))  
        # print('o4 =', o4)
        return o4

class EarlyStop():
    def __init__(self, size): # size = buffer_size = 30
        self.img_collection = []
        self.size = size

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img) # Adds an element to the end of the List
        # print('self.img_collection =', self.img_collection)
        # print('self.img_collection length =', len(self.img_collection))
        # print('self.img_collection[0] =', self.img_collection[0])
        if len(self.img_collection) > self.size:
            self.img_collection.pop(0) # Removes img_collection[0]
    def get_img_collection(self):
        return self.img_collection
    def Var_cal(self, x1, x2): # Add 'self'
        # print('x1 =', x1)
        # print('x2 =', x2)
        # a = x1-x2
        # print('x1-x2 =', a)
        # b = a**2
        # print('(x1-x2)**2 =', b)
        # c = b.sum()
        # print('((x1-x2)**2).sum() =', c)
        # d = x1.size
        # print('x1.size =', x1.size)
        # e = c/d
        # print('((x1 - x2) ** 2).sum() / x1.size =', e)
        # return e
        return ((x1 - x2) ** 2).sum() / x1.size
    
# Reset the Weight of the convolutional or linear layers to random values
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters() 
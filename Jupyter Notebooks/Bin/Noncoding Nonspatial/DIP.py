import numpy as np
import torch
from torch import nn



class DeepImagePrior(object):
    def __init__(self,user_num,M, iteration,LR,buffer_size,threshold,stop):
        
        self.user_num = user_num    ####number of transmitted symbol in real domain
        self.M = M                  ####modulation order, 4 for 4qam, 16 for 16qam
        self.iteration = iteration  ####number of max iterations used for DIP
        self.LR = LR                ####Learning rate,  typically set to 0.01
        self.buffer_size = buffer_size    ###iterations stored,  typically set to 30
        self.threshold = threshold        ###Threshold of DIP stop,, typically set to 0.001
        self.stop = stop                  ###True
        constellation = np.linspace(int(-np.sqrt(M) + 1), int(np.sqrt(M) - 1), int(np.sqrt(M)))
        alpha = np.sqrt((constellation ** 2).mean())
        constellation /= (alpha * np.sqrt(2))
        self.constellation = constellation
        constellation_expanded = np.expand_dims(self.constellation, axis=1)
        constellation_expanded= np.repeat(constellation_expanded[None,...],1,axis=0)
        
        constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), self.user_num, axis=1)
        
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
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        dtype = torch.FloatTensor
        batch_size = H.shape[0]
        x_dip_ay = np.empty((batch_size,self.user_num))
        num_stop_point = []
        
        for bs in range(batch_size):

            i = 0            
            flag = False
            
            net = Decoder(self.user_num).type(dtype)    ###Define the Neural network
            mse = torch.nn.MSELoss().type(dtype)        ###Loss function
            
            optimizer = torch.optim.Adam(net.parameters(), lr= self.LR)     ###Adam optimizer
            net.apply(weight_reset)                     ###Reset the neural network parameters
            net_input = torch.randn(1,4)                ###Random input for DIP
            
            y_torch = torch.from_numpy(Y[bs]).reshape(1,1,self.user_num,1).type(dtype)
    
            H_torch = torch.from_numpy(H[bs]).reshape(1, 1, self.user_num,self.user_num).type(dtype)
            
            variance_history = []
            earlystop = EarlyStop(size= self.buffer_size)
            
            while i < self.iteration and flag==False :
                
                i+=1 
                
                # step 1, update network:
                optimizer.zero_grad()
                out = net(net_input).type(dtype)*np.max(self.constellation)
                out1 = out.reshape(1,1,self.user_num,1).type(dtype)
                Y_hat = torch.matmul(H_torch,out1).type(dtype)
                total_loss = mse(Y_hat,y_torch)
                total_loss.backward()
                optimizer.step()

                x_dip = out.detach().cpu().numpy()
               
                if  self.stop is True:
                    r_img_np = x_dip.reshape(-1)
                    earlystop.update_img_collection(r_img_np)
                    img_collection = earlystop.get_img_collection()
                    if len(img_collection) ==  self.buffer_size:
                        ave_img = np.mean(img_collection, axis=0)
                        variance = []
                        for tmp in img_collection:
                            variance.append(earlystop.Var_cal(ave_img, tmp))
                        cur_var = np.mean(variance)
                        variance_history.append(cur_var)
                    else:
                        cur_var = 0
                    
                    if cur_var != 0 and cur_var <  self.threshold:
                        num_stop_point.append(i)
                        flag = True                
                        
            x_dip_ay[bs] = x_dip.reshape(-1)
        
        return x_dip_ay,num_stop_point

class Decoder(nn.Module):
    def __init__(self,user_num):
        super().__init__()
        
        self.nn1 = nn.Linear(4,8)
        self.nn2 = nn.Linear(8,16)
        self.nn3 = nn.Linear(16,32)
        self.nn4 = nn.Linear(32,user_num)
        self.act = nn.Tanh()
         
    def forward(self,x): 
        o1 = self.act(self.nn1(x)) 
        o2 = self.act(self.nn2(o1)) 
        o3 = self.act(self.nn3(o2)) 
        o4 = self.act(self.nn4(o3))  
          
        return o4    

class EarlyStop():
    def __init__(self, size):
        self.img_collection = []
        self.size = size

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.size:
            self.img_collection.pop(0)
    def get_img_collection(self):
        return self.img_collection
    def Var_cal(x1, x2):
        return ((x1 - x2) ** 2).sum() / x1.size
    
    
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters() 
        
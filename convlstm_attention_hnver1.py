import torch
import numpy as np
import pandas as pd 
from dataset_preprocessing import data_loading_for_years, torch_data 
from glob import glob 
from tqdm import tqdm 
from torch import optim 
import torch.nn as nn 
#from feature_loss import SSIM

#문제 정의 : 12 to 12 : 12주 보고 12주 예측 
class self_attention(nn.Module): #self attention class
    def __init__(self,input_dim, hidden_dim, device):
        super().__init__()
        #conv layer로 q, k, v를 구합니다. 
        #논문에 따라 filter size = (1,1)
        self.layer_q = nn.Conv2d(input_dim, hidden_dim ,1)#, stride = 1, padding = 1) # 1X1 conv
        self.layer_k = nn.Conv2d(input_dim, hidden_dim,1)#, 1, padding = 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_f = nn.Conv2d(input_dim, input_dim, 1) 
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
    def forward(self, q, k ,v ,mask = None):
        batch_size, channel, H, W = q.shape
        h = q #shart connection을 위해서 원본 q 저장. 
        
        q = self.layer_q(q)
        k = self.layer_k(k)
        v = self.layer_v(v)
        q = q.view(batch_size, self.hidden_dim, H*W )
        q = q.transpose(1,2)
        k = k.view(batch_size,  self.hidden_dim, H*W )
        v = v.view(batch_size,  self.input_dim, H*W )

        e = torch.bmm(q,k) 
        #print('e shape is', e.shape)
        attention = torch.softmax(e, dim = -1) #attention을 구해줍니다. 
        #print('attention shape is', attention.shape) # batch_size, H*W, H*W
        z = torch.matmul(attention, v.permute(0,2,1)) #value * attention
        #print('z shape is', z.shape)
        z = z.view(batch_size, self.input_dim, H, W)
        Z = z * h 
        #print('Z shape is', Z.shape)
        #short cut connection 
        out = self.layer_f(z*h) + h

        return out, attention



class convlstm_cell(nn.Module):
    def __init__(self, params):
        super(convlstm_cell, self).__init__()
        self.input_channels = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.kernel_size= params['kernel_size']
        self.padding = params['padding']
        self.device = params['device']
        self.attention_x = self_attention(params['input_dim'], params['att_hidden_dim'], self.device)
        self.attention_h = self_attention(params['hidden_dim'], params['att_hidden_dim'], self.device )
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = self.input_channels + self.hidden_dim, out_channels = 4 * self.hidden_dim, kernel_size= self.kernel_size, padding = self.padding)
                     ,nn.GroupNorm(4* self.hidden_dim, 4* self.hidden_dim ))  #groupnorm 사용   
    def forward(self, x, hidden):
        h, c = hidden
        #x 와 h를 self attention으로 구해주기. 
        x, _ = self.attention_x(x, x, x)
        hidden, _ = self.attention_h(h, h, h)
        combined = torch.cat([x, hidden], dim = 1) #combine 해서 (batch_size, input_dim + hidden_dim, img_size[0], img_size[1])
        combined_conv = self.conv2d(combined)# conv2d 후 (batch_size, 4 * hidden_dim, img_size[0], img_size[1])
        i, f, o ,g =torch.split(combined_conv, self.hidden_dim, dim =1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i*g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)
    
    def init_hidden(self, batch_size, img_size):
        h, w = img_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
               torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device))


class Sa_convlstm(nn.Module): #base model
    def __init__(self, params):
        super(Sa_convlstm, self).__init__()
        #hyperparameters
        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.cells, self.bns = [], []
        self.n_layers = params['n_layers']
        self.input_window_size = params['input_window_size']
        #layer loading
        for i in range(params['n_layers']):
            params['input_dim'] = params['input_dim'] if i == 0 else params['hidden_dim']
            params['hidden_dim'] = params['hidden_dim'] if i != params['n_layers']-1 else 1 
            self.cells.append(convlstm_cell(params))
            self.bns.append(nn.LayerNorm((params['hidden_dim'], 56, 38))) #normalization 방법으로 layernorm 사용
        self.cells = nn.ModuleList(self.cells)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x, hidden = None):
        if hidden == None:
            hidden = self.init_hidden(batch_size = self.batch_size, img_size = self.img_size)
        
        input_x = x
        for i, layer in enumerate(self.cells):
            out_layer = []
            hid = hidden[i]
            for t in range(self.input_window_size):
                input_x = x[:,t,:,:,:] #input window size에서 하나씩.
                out, hid = layer(input_x, hid)
                out = self.bns[i](out)
                out_layer.append(out)
            out_layer = torch.stack(out_layer, dim = 1)
            x = out_layer
        return out_layer 

    def init_hidden(self, batch_size, img_size): #hidden state initialize 
        states = [] 
        for i in range(self.n_layers):
            states.append(self.cells[i].init_hidden(batch_size, img_size))
        return states 

class Model():
    def __init__(self, params, loading_path = None, set_device = 4):
        if loading_path:
            self.model = torch.load(loading_path, map_location = 'cuda:{}'.format(set_device)).to(params['device'])
        else:
            self.model = Sa_convlstm(params).to(params['device'])
        self.criterion = nn.MSELoss()
        self.output = params['output_dim']
        self.device = params['device']
        self.optim = optim.Adam(self.model.parameters(), lr = params['lr']) 
    def train(self, train_dataset, epochs, path = './model_save/best_sa_aft100epochs.pth'):
        for i in range(epochs):
            losses = [] 
            for _, data in enumerate(train_dataset):
                x , y = data 
                x= x.to(self.device)
                y =y.to(self.device)
                self.optim.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)#(pred[:,-self.output:, :,:,:], y)
                loss.backward()
                self.optim.step()
                losses.append(loss.data.cpu().numpy())
            print('{}th epochs loss {}'.format(i, np.mean(losses)))
            torch.save(self.model, path)


    # def evaluate(self, test_dataset,ver = 1): 
    #     prediction = []
    #     for i, data in enumerate(test_dataset):
    #         x,y = data 
    #         x = x.to(self.device)
    #         if i < 32 and i !=0:
    #             self.model.train()
    #             self.optim.zero_grad()
    #             loss = self.criterion(pred,x)
    #             loss.backward()
    #             self.optim.step()
    #         else:
    #             self.model.eval()
    #         pred = self.model(x)
    #         prediction.extend(pred)
                
    #     prediction = torch.stack(prediction, dim = 1).detach().cpu()
    #     np.save('./test_results_12to12_layernorm.npy'.format(ver), prediction.detach().cpu().numpy())
    #     return prediction 

if __name__ == '__main__':
    if torch.cuda.is_available() :
        device = torch.device('cuda')
    else : 
        device = torch.device('cpu')
    torch.cuda.empty_cache() #메모리 없애는 용도
    set_device = 4
    torch.cuda.set_device(set_device)
    print(set_device)
    torch.manual_seed(42)
    BATCH_SIZE= 8 
    img_size = (448,304)
    new_size = (56, 38)
    input_window_size, output  = 12,12
    epochs = 1000
    lr = 1e-3
    weeks = 24*3
    loss = 'L1'#'L1+L2'#'L1' #L2 
    ##########################################################
    data_class = data_loading_for_years() #년별 추론용.
    data_class.data_load('./data/weekly_train/*.npy', years_pre = weeks) 
    data_class.zero_padding()
    train_D, train_L= data_class.preprocessing_and_split_and_resize(input_window_size, output, stride= new_size, img_size= new_size, split_bool = False)
    img_size = (train_D.shape[2], train_D.shape[3])
    data_tf = torch_data(BATCH_SIZE)
    att_hid_dim = 16
    train_data = data_tf.torch_data(train_D, train_L, False, True)
    data_class.test_load('./data/weekly_train/*.npy', length = 24)
    test_D = data_class.preprocessing_and_split_and_resize(input_window_size, 0, stride= new_size, img_size= new_size, split_bool = False, TRAIN = False) 
    test_data = data_tf.torch_data(test_D, test_D, False, True)
    #valid_data = data_tf.torch_data(valid_D, valid_L, True, True)
    ################# 여기까지가 data preprocessing ###########

    params= {'input_dim': 1 ,'batch_size' : BATCH_SIZE, 'padding':1, 'lr' : lr, 'device':device, 'att_hidden_dim':att_hid_dim, 'kernel_size':3, 'img_size':img_size, 'hidden_dim': 16 , 'n_layers': 3, 'output_dim': output, 'input_window_size':input_window_size, 'loss':loss}
    model = Model(params)#, loading_path = './model_save/SA_12to12_BS8_1000epochs_72weeks_layer_norm_att_hid_dim16.pth') 
    model.train(train_data, epochs = epochs, path = './model_save/SA_{}to{}_BS{}_{}epochs_{}weeks_layer_norm_att_hid_dim{}_{}loss.pth'.format(input_window_size, output, BATCH_SIZE, epochs, weeks, att_hid_dim)) 
    prediction = model.evaluate(test_data)#, set_device = set_device, path = './model_save/bast_sa.pth')#SA_{}to{}_BS{}.pth'.format(input_window_size, output, BATCH_SIZE) , set_device= set_device)
    #print(prediction.shape)
    prediction = prediction.transpose(0,1)
    prediction = prediction.reshape(16,8,12,1,new_size[0], new_size[1])
    #print(type(prediction))
    # prediction = np.where(prediction>1, 1. , prediction)
    # prediction = np.where(prediction<0, 0. , prediction)
   #att hid dim 1 with 1000 epochs -> 0.0007 loss
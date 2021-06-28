import torch
import numpy as np
import pandas as pd 
from dataset_preprocessing import data_loading_for_years, torch_data 
from glob import glob 
from tqdm import tqdm 
from torch import optim 
import torch.nn as nn 
from feature_loss import SSIM
#from feature_loss import SSIM

class self_attention_memory_module(nn.Module): #SAM 
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        # h(hidden)를 위한 layer q, k, v 
        # m(memory)을 위한 layer k2, v2 
        #layer z, m은 attention_h와 attention_m concat 후의 layer.  
        self.layer_q = nn.Conv2d(input_dim, hidden_dim ,1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim,1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim,1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_z = nn.Conv2d(input_dim *2, input_dim*2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        #feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H*W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H*W)
        Q_h = Q_h.transpose(1, 2)
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim = -1) #batch_size, H*W, H*W
        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H*W)
        Z_h = torch.matmul(A_h, V_h.permute(0,2,1))

        ###### memory m attention #####
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H*W)
        V_m = V_m.view(batch_size, self.input_dim, H*W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim = -1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0,2,1))
        Z_h = Z_h.transpose(1,2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1,2).view(batch_size, self.input_dim, H, W)

        ### attention으로 구한 Z_h와 Z_m concat 후 계산 ####
        W_z = torch.cat([Z_h , Z_m], dim = 1)
        Z = self.layer_z(W_z)
        ## Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim = 1)) # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim = 1)
        ### 논문의 수식과 같습니다(figure)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m 

        return  new_h, new_m 
        
class SA_Convlstm_cell(nn.Module):
    def __init__(self, params):
        super().__init__()
        #hyperparrams 
        self.input_channels = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.kernel_size= params['kernel_size']
        self.padding = params['padding']
        self.device = params['device']
        self.attention_layer = self_attention_memory_module(params['hidden_dim'], params['att_hidden_dim'], self.device)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = self.input_channels + self.hidden_dim, out_channels = 4 * self.hidden_dim, kernel_size= self.kernel_size, padding = self.padding)
                     ,nn.GroupNorm(4* self.hidden_dim, 4* self.hidden_dim ))   #(num_groups, num_channels)     

    def forward(self, x, hidden):
        h, c, m = hidden
        combined = torch.cat([x, h], dim = 1) #combine 해서 (batch_size, input_dim + hidden_dim, img_size[0], img_size[1])
        combined_conv = self.conv2d(combined)# conv2d 후 (batch_size, 4 * hidden_dim, img_size[0], img_size[1])
        i, f, o ,g =torch.split(combined_conv, self.hidden_dim, dim =1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i*g
        h_next = o * torch.tanh(c_next)
        #### 위까지는 보통의 convlstm과 같음. 
        ### attention 해주기 
        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, (h_next, c_next, m_next)
    
    def init_hidden(self, batch_size, img_size): #h, c, m initalize
        h, w = img_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
               torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
               torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device))

class Sa_convlstm(nn.Module): #self-attention convlstm for spatiotemporal prediction model
    def __init__(self, params):
        super(Sa_convlstm, self).__init__()
        #hyperparams 

        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.cells, self.bns = [], []
        self.n_layers = params['n_layers']
        self.input_window_size = params['input_window_size']
        for i in range(params['n_layers']):
            params['input_dim'] = params['input_dim'] if i == 0 else params['hidden_dim']
            params['hidden_dim'] = params['hidden_dim'] if i != params['n_layers']-1 else 1 
            self.cells.append(SA_Convlstm_cell(params))
            self.bns.append(nn.LayerNorm((params['hidden_dim'], 56, 38))) #layernorm 사용
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
                input_x = x[:,t,:,:,:]
                out, hid = layer(input_x, hid)
                out = self.bns[i](out)
                out_layer.append(out)
            out_layer = torch.stack(out_layer, dim = 1) #output predictions들 저장. 
            x = out_layer
        return out_layer 

    def init_hidden(self, batch_size, img_size):
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
        self.loss = params['loss']
        if self.loss == 'SSIM':
            self.criterion = SSIM().to(device)
        elif self.loss == 'L2':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()
        self.output = params['output_dim']
        self.device = params['device']
        self.optim = optim.Adam(self.model.parameters(), lr = params['lr']) 
    def train(self, train_dataset, epochs, path = './model_save/best_sa_aft100epochs.pth'):
        min_loss = 1e9
        for i in range(epochs):
            losses, val_losses = [], [] 
            self.model.train() 
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
            ##valid 있이 하고, valid loss 에따라 모델 저장을 할때는 이 주석을 풀면 됩니다. def train(self, train_dataset 옆에 , valid_dataset을 추가해주세요.)
            # self.model.eval()
            # with torch.no_grad():
            #     for _, data in enumerate(valid_dataset):
            #         x, y=  data 
            #         x= x.to(self.device)
            #         y= y.to(self.device)
            #         pred = self.model(x)
            #         loss = self.criterion(pred, y)
            #         val_losses.append(loss.data.cpu().numpy())
            # print('{}th epochs loss {}, valid loss {}'.format(i, np.mean(losses), np.mean(val_losses)))
            # if np.mean(val_losses) < min_loss:
            #     torch.save(self.model, path)
            #     min_loss = np.mean(val_losses)
            print(np.mean(losses))
            torch.save(self.model, path)


    def evaluate(self, test_dataset, path = './SACL_test_results_real_12to12_layernorm.npy'):
        prediction = [] 
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(test_dataset):
                x, y = data 
                x = x.to(self.device)
                pred = self.model(x)
                prediction.extend(pred)
        prediction = torch.stack(prediction, dim = 1) 
        np.save(path, prediction.detach().cpu().numpy())
        return prediction
          

        



if __name__ == '__main__':
    if torch.cuda.is_available() :
        device = torch.device('cuda')
    else : 
        device = torch.device('cpu')
    torch.cuda.empty_cache() #메모리 없애는 용도
    set_device = 5
    torch.cuda.set_device(set_device)
    print(set_device)
    torch.manual_seed(42)
    BATCH_SIZE= 8 
    img_size = (448,304)
    new_size = (56, 38)
    strides = img_size 
    input_window_size, output  = 12,12
    epochs = 1000
    lr = 1e-3
    weeks = 24*2
    hid_dim = 32
    #SSIM 쓰면 너무 patch별로 차이가 뚜렷해짐 -> 안쓰는게 나을듯 
    loss = 'L2'#'L1+L2'#'L1' #L2 
    #L1, L2, SSIm 중에서는 L2(MSE) 가 제일 좋은듯 
    data_class = data_loading_for_years() #년별 추론용.
    data_class.data_load('./data/weekly_train/*.npy', years_pre = weeks) 
    data_class.zero_padding()
    train_D, train_L= data_class.preprocessing_and_split_and_resize(input_window_size, output, stride= new_size, img_size= new_size, split_bool = False)
    img_size = (train_D.shape[2], train_D.shape[3])
    data_tf = torch_data(BATCH_SIZE)
    att_hid_dim = 16 #16 #더 작아지면 오히려 안좋은듯? 64 에서도 좋은 성과 X 아님 layer 문제
    #layer 1, 2, 3, 4 했을 때 3개가 제일 나음.
    train_data = data_tf.torch_data(train_D, train_L, True, True)
    data_class.test_load('./data/weekly_train/*.npy', length = 24)
    test_D = data_class.preprocessing_and_split_and_resize(input_window_size, 0, stride= new_size, img_size= new_size, split_bool = False, TRAIN = False) 
    test_data = data_tf.torch_data(test_D, test_D, False, True)
    #valid_data = data_tf.torch_data(valid_D, valid_L, True, True)
    #config = {'lr': tune.loguniform(1e-4, 1e-2), 'hidden_dim': tune.uniform(1, 512), 'att_hid_dim': tune.uniform(1, 512)}
    params= {'input_dim': 1 ,'batch_size' : BATCH_SIZE, 'padding':1, 'lr' : lr, 'device':device, 'att_hidden_dim':att_hid_dim, 'kernel_size':3, 'img_size':img_size, 'hidden_dim': hid_dim , 'n_layers': 3, 'output_dim': output, 'input_window_size':input_window_size, 'loss':loss}
    model = Model(params)#, loading_path = './model_save/SAConvLSTM_12to12_BS8_100epochs_72weeks_8loss_4layers.pth') 
    model.train(train_data, epochs = epochs, path = './model_save/SAConvLSTM_{}to{}_BS{}_{}epochs_{}weeks_{}atthid_{}loss_3layers_{}hid_no_valid.pth'.format(input_window_size, output, BATCH_SIZE, epochs, weeks, att_hid_dim, loss, hid_dim)) 
    prediction = model.evaluate(test_data, path = './imgs/AConvLSTM_{}to{}_BS{}_{}epochs_{}weeks_{}atthid_{}loss_3layers_{}hid_no_valid.npy'.format(input_window_size, output, BATCH_SIZE, epochs, weeks, att_hid_dim, loss, hid_dim)) 

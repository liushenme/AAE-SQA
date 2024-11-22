import json
import os
import math
from einops import repeat
from scipy.stats import pearsonr
import fairseq

import numpy as np
np.set_printoptions(threshold=np.inf)

import torch
from torch import nn
import torch.nn.functional as F

from asteroid.engine.system import System
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import magreim, mag
from asteroid_filterbanks.transforms import apply_real_mask
from asteroid_filterbanks.transforms import apply_mag_mask
from asteroid_filterbanks.transforms import from_magphase, from_torch_complex, magphase
from asteroid.masknn.recurrent import StackedResidualRNN
from asteroid.engine.optimizers import make_optimizer
from asteroid import torch_utils
from torch.nn.utils import weight_norm

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

torch.set_printoptions(profile="full")

Tensor = TypeVar('torch.tensor')

from abc import abstractmethod

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):     
        x = torch.Tensor(x) 
    elif 'int' in str(x.dtype):    
        x = torch.LongTensor(x)
    else:       
         # raise Exception("Error!")
        return x      
    return x.to(device)

def e2e_make_model_and_optimizer(conf):
    # Define building blocks for local model
    stft, istft = make_enc_dec("stft", **conf["filterbank"])

    if conf["main_args"]["model_type"] != None:
        model_type = conf["main_args"]["model_type"]
        Masker = eval(model_type)
        masker = Masker()

    model = Model_e2e_2input(stft, masker)
    optimizer = make_optimizer(filter(lambda p: p.requires_grad, model.parameters()), **conf["optim"])
    return model, optimizer

class Model_e2e(nn.Module):

    def __init__(self, encoder, masker):
        super().__init__()
        self.encoder = encoder
        self.masker = masker

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # Compute STFT
        #print('x=',x.size())
        tf_rep = self.encoder(x)
        tf_rep = mag(tf_rep)
        
        mos = self.masker(tf_rep) # mask is real (bs, f, t)
        #print("est_masks=",est_masks.size())
        # Apply TF mask
        return mos

class MOSNet(nn.Module):
    def __init__(self, activation = nn.ReLU):
        super(MOSNet, self).__init__()
        self.mean_net_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),
            activation(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            activation(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            activation(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            activation()
        )

        self.mean_net_rnn = nn.LSTM(input_size = 512, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.mean_net_dnn = nn.Sequential(
            nn.Linear(256, 128),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            activation()
        )

    def forward(self, spectrum):
        #spectrum should have shape (batch, 1, time, 257)
        spectrum = spectrum.transpose(1, 2)
        spectrum = spectrum.unsqueeze(1)
        batch = spectrum.shape[0]
        time = spectrum.shape[2]
        mean_feat = self.mean_net_conv(spectrum)
        #print(mean_feat.shape)
        mean_feat = mean_feat.permute(0, 2, 1, 3).contiguous()  # x becomes (batch, time, channel, frequency)
        mean_feat = mean_feat.view((-1, mean_feat.size(1), mean_feat.size(2) * mean_feat.size(3)))
        #mean_feat = mean_feat.view((batch, time, 256))
        mean_feat, (h, c) = self.mean_net_rnn(mean_feat)
        mean_feat = self.mean_net_dnn(mean_feat)
        mean_feat = mean_feat.squeeze(-1)
        mean_scores = torch.mean(mean_feat, dim = -1)
        #return mean_scores, mean_feat
        return mean_scores




class Model_e2e_2input(nn.Module):

    def __init__(self, encoder, masker):
        super().__init__()
        self.encoder = encoder
        self.masker = masker

    #def forward(self, x, judge_id):
    def forward(self, *args):
        x = args[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if len(args) == 2 :
            judge_id = args[1]
            tf_rep = self.encoder(x)
            tf_rep = mag(tf_rep)
            mos = self.masker(tf_rep, judge_id) # mask is real (bs, f, t)
        else:
            tf_rep = self.encoder(x)
            tf_rep = mag(tf_rep)
            mos = self.masker(tf_rep) # mask is real (bs, f, t)
        #print("est_masks=",est_masks.size())
        # Apply TF mask
        return mos

class aae_3t_deep(nn.Module):
    def __init__(self):
        
        super(aae_3t_deep, self).__init__()

        #self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=(1, 3), padding = (1, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 3), padding = (1, 1))
        self.conv3 = nn.Conv2d(32, 64, (3, 3), stride=(1, 3), padding = (1, 1))
        self.conv4 = nn.Conv2d(64, 128, (3, 3), stride=(1, 3), padding = (1, 1))
        #self.conv5 = nn.Conv2d(128, 256, (3, 3), stride=(1, 3), padding = (1, 1))
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        #self.bn5 = nn.BatchNorm2d(256)
        
        #self.debn5 = nn.BatchNorm2d(128)
        self.debn4 = nn.BatchNorm2d(64)
        self.debn3 = nn.BatchNorm2d(32)
        self.debn2 = nn.BatchNorm2d(16)
        self.debn1 = nn.BatchNorm2d(1)


        #self.deconv5 = nn.ConvTranspose2d(512, 128, (3, 3), stride=(1, 3), padding = (1, 1))
        self.deconv4 = nn.ConvTranspose2d(128, 64, (3, 3), stride=(1, 3), padding = (1, 1))
        self.deconv3 = nn.ConvTranspose2d(64, 32, (3, 3), stride=(1, 3), padding = (1, 1), output_padding = (0,1))
        self.deconv2 = nn.ConvTranspose2d(32, 16, (3, 3), stride=(1, 3), padding = (1, 1), output_padding = (0,1))
        self.deconv1 = nn.ConvTranspose2d(16, 1, (3, 3), stride=(1, 3), padding = (1, 1), output_padding = (0,1))

        
        self.mos_pred = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            nn.ReLU()
        )
        
        self.cla_cla = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 500),
        )
        
        self.spk_cla = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,60),
        )
        #self.cla_softmax= nn.LogSoftmax(dim=1)
        #self.spk_softmax= nn.LogSoftmax(dim=1)
        #self.LogSoftmax= nn.Softmax(dim=1)

    def forward(self, input, a):
        """Input: (batch_size, times_steps, freq_bins)"""
        x = input 
        x = x.transpose(1, 2)
        
        #x = x.transpose(1, 2)
        #print(x.shape)
        origin = x
        x = x[:, None, :, :]    
        x1 = F.relu_(self.bn1(self.conv1(x)))
        #print(x1.shape)
        x2 = F.relu_(self.bn2(self.conv2(x1)))
        #print(x2.shape)
        x3 = F.relu_(self.bn3(self.conv3(x2)))
        #print(x3.shape)
        x4 = F.relu_(self.bn4(self.conv4(x3)))
        #print(x4.shape)
        
        dex4 = F.relu_(self.debn4(self.deconv4(x4)))
        #print(dex4.shape)
        dex3 = F.relu_(self.debn3(self.deconv3(dex4)))
        #print(dex3.shape)
        dex2 = F.relu_(self.debn2(self.deconv2(dex3)))
        #print(dex2.shape)
        dex1 = F.relu_(self.debn1(self.deconv1(dex2)))
        #print(dex1.shape)

        out = dex1.squeeze(1)
        # = framewise_output.squeeze()
        feat = x4
        x4 = x4.transpose(1, 2)

        x4 = x4.reshape((-1, x4.size(1), x4.size(2) * x4.size(3)))

        #feat = x4
        mean_score = self.mos_pred(x4)
        mean_score = mean_score.squeeze(-1)
        mean_score = torch.mean(mean_score, dim = 1)
        
        x4 = torch.mean(x4, dim = 1)
        
        mean_class = self.cla_cla(x4)
        
        reverse_feature = ReverseLayerF.apply(x4, a)
        
        speaker = self.spk_cla(reverse_feature)
        #print(self.mos_pred[0].weight)

        return out, origin, feat, mean_score, speaker, mean_class


class cnn_lstm_4d_fu(nn.Module):
    def __init__(self, activation = nn.ReLU):
        super(cnn_lstm_4d_fu, self).__init__()
        self.mean_net_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.BatchNorm2d(16),
            activation(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.BatchNorm2d(32),
            activation(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.BatchNorm2d(64),
            activation(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.BatchNorm2d(128),
            activation(),
            nn.Dropout(0.3),
        )

        self.mean_net_rnn = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 1, batch_first = True, bidirectional = True)
        self.mean_net_dnn = nn.Sequential(
            #nn.Linear(512, 512),
            #activation(),
            #nn.Dropout(0.5),
            nn.Linear(512, 128),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            activation()
        )
        #self.rc = nn.Linear(1024, 512)
        self.rc = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 1, padding = 0, stride = 1)

    def forward(self, spectrum, latent_feat):
        #spectrum should have shape (batch, 1, time, 257)
        spectrum = spectrum.transpose(1, 2)
        spectrum = spectrum.unsqueeze(1)
        batch = spectrum.shape[0]
        time = spectrum.shape[2]
        mean_feat = self.mean_net_conv(spectrum)
        
        latent_feat = latent_feat.squeeze(1).transpose(1, 2)

        fu_feat = torch.cat((mean_feat, latent_feat), 1)
        fu_feat = self.rc(fu_feat)
        fu_feat = fu_feat.permute(0, 2, 1, 3).contiguous()  # x becomes (batch, time, channel, frequency)
        fu_feat = fu_feat.view(batch, time, 512)
    
        fu_feat, (h, c) = self.mean_net_rnn(fu_feat)
        fu_feat = self.mean_net_dnn(fu_feat)
        fu_feat = fu_feat.squeeze(-1)
        frame_scores = fu_feat
        mean_scores = torch.mean(fu_feat, dim = -1)

        return mean_scores, frame_scores


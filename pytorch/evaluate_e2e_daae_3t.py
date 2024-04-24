import numpy as np
import logging
from sklearn import metrics
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from pypesq import pesq


import torch
import sys
import os
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

from data_generator import DAAEDataset_test_3t
#from model2 import *
from model import e2e_load_best_model

import argparse
import yaml
import json
from pprint import pprint

from asteroid.metrics import get_metrics
from torch.utils.data import DataLoader, random_split
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument("--json_dir", type=str, required=True, help="Test directory including wav files"
)


def collate_fn_pad(batch):     
    """
    Returns:
       [B, F, T (Longest)]     
    """
    noisy_list = []
    target_list = []           
    name_list = []           
    spk_list = []           
    cla_list = []           

    for noisy, target, name, spk, cla in batch:
        noisy_list.append(noisy)  # [F, T] => [T, F]
        #clean_list.append(clean)  # [1, T] => [T, 1]
        #n_frames_list.append(n_frames) 
        target_list.append(target)      
        name_list.append(name)      
        spk_list.append(spk)      
        cla_list.append(cla)      
    noisy_list = pad_sequence(noisy_list, batch_first=True)  # ([T1, F], [T2, F], ...) => [T, B, F] => [B, F, T]
    target_list = torch.tensor(target_list)
    spk_list = torch.tensor(spk_list)
    cla_list = torch.tensor(cla_list)
  
    return noisy_list, target_list, name, spk_list, cla_list


def main(conf):
    # Get best trained model
    #for test_dir in ["11_2","04", "13","11_1","14_2","17","val","test"]:
    #for test_dir in ["04","17"]:
    #for test_dir in ["04","18"]:
    #for test_dir in ["02","03","04","05","06","07_1","07_2","08","09","11_1","11_2","11_3","13","14_1","14_2","17","18","val","test"]:
    #for test_dir in ["01", "02", "05", "08", "13_1", "13_2", "14_1", "14_2", "17", "train", "val", "test"]:
    print(conf)
    exp_dir = conf["exp_dir"]
    feat_dir = os.path.join(exp_dir, 'feat')
    print(feat_dir)
    os.makedirs(feat_dir, exist_ok=True)
    for test_dir in ["train", "val", "test"]:
        test_set = DAAEDataset_test_3t(os.path.join(conf["json_dir"], test_dir))
        test_loader = DataLoader(
            test_set,
            shuffle=False,
            #batch_size=conf["train_conf"]["training"]["batch_size"],
            batch_size=1,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_fn_pad
        )

        model = e2e_load_best_model(conf["train_conf"], conf["exp_dir"])
        if conf["use_gpu"]:
            model = model.cuda()
        # Evaluate performances separately w/ and w/o reverb
        evaluate(model, test_loader, feat_dir)
    print('finish')

def evaluate(model, data_loader, feat_dir):

    output_dict = forward(
        model=model, 
        data_loader=data_loader, 
        feat_dir=feat_dir,
        return_input=False, 
        return_target=True)

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        # raise Exception("Error!")
        return x

    return x.to(device)

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

def forward(model, data_loader, feat_dir, return_input=False, return_target=False):
    """Forward data to model.

    Args:
      model: object
      generator: object
      return_input: bool
      return_target: bool

    Returns:
      output_dict: {'audio_name': (N,)
                    'clipwise_output': (N, classes_num), 
                    'framewise_output': (N, frames_num, classes_num), 
                    (optional) 'target': (N, classes_num), 
                    (optional) 'strong_target': (N, frames_num, classes_num)}
    """

    device = next(model.parameters()).device #see model device
    t_time = 0
    wav_time = 0
    # Evaluate on mini-batch
    for n, wav_dic in enumerate(data_loader):
        #print(batch_data_dict['audio_name'].shape)
        # Predict
        #print(wav_dic)
        (wav, target, name, _, len_x) = wav_dic
        wav_time += len_x
        #print("label : " , target)
        batch_waveform = move_data_to_device(wav, device)
        
        with torch.no_grad():
            model.eval()
            #batch_output, _, feats = model(batch_waveform)
            #batch_output, _, _, _, feats = model(batch_waveform)
            s_t = time.time()
            out = model(batch_waveform, 1)
            i_t = time.time() - s_t
            print(i_t)
            batch_output = out[1]
            feats = out[-4]
        np.save(os.path.join(feat_dir, str(name) + '.npy'), feats.cpu().numpy())
        t_time = t_time + i_t
        #torch.save(feats, os.path.join(feat_dir, str(name) + '.pt'))
        #batch_output = torch.clamp(batch_output, 1, 5)
        #print("predi : " ,batch_output)
    print('time', t_time / wav_time)

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as conf_file:
        train_conf = yaml.safe_load(conf_file)
    arg_dic["train_conf"] = train_conf
    pprint(arg_dic)
    main(arg_dic)

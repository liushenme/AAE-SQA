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

from data_generator import FUDataset
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
parser.add_argument("--feat_dir", default="exp/tmp", help="Experiment root")
parser.add_argument("--json_dir", type=str, required=True, help="Test directory including wav files"
)



def collate_fn_pad(batch):     
    """
    Returns:
       [B, F, T (Longest)]     
    """
    noisy_list = []
    #clean_list = []
    target_list = []     
    feat_list = []
    name_list = []

    for noisy, target, feat, name in batch:
        noisy_list.append(noisy)  # [F, T] => [T, F]
        target_list.append(target)      
        feat_list.append(feat)
        name_list.append(name)
    noisy_list = pad_sequence(noisy_list, batch_first=True)  # ([T1, F], [T2, F], ...) => [T, B, F] => [B, F, T]
    target_list = torch.tensor(target_list)
    feat_list = pad_sequence(feat_list, batch_first=True)
    
    return noisy_list, target_list, feat_list, name_list

def main(conf):
    # Get best trained model
    #for test_dir in ["11_2","04", "13","11_1","14_2","17","val","test"]:
    #for test_dir in ["04","17"]:
    #for test_dir in ["04","18"]:
    #for test_dir in ["02","03","04","05","06","07_1","07_2","08","09","11_1","11_2","11_3","13","14_1","14_2","17","18","val","test"]:
    #for test_dir in ["01", "02", "05", "08", "13_1", "13_2", "14_1", "14_2", "17", "train", "val", "test"]:
    for test_dir in ["train", "val", "test"]:
    #for test_dir in ["17"]:
        test_set = FUDataset(os.path.join(conf["json_dir"], test_dir), conf["feat_dir"])
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
        (statistics, output_dict) = evaluate(model, test_loader)
        print('{} statistics:'.format(test_dir))
        print('    rmse : {:.3f}'.format(np.mean(statistics['mse'])))
        print('    pearson_s : {:.3f}'.format(np.mean(statistics['pearson_s'])))
        print('    SRCC : {:.3f}'.format(np.mean(statistics['SRCC'])))
        print('     p_value: {:f}'.format(np.mean(statistics['p_value'])))
        

def evaluate(model, data_loader):

    output_dict = forward(
        model=model, 
        data_loader=data_loader, 
        return_input=False, 
        return_target=True)
        
    #print(output_dict.shape)

    statistics = {}

    isnan =  (True in np.isnan(output_dict['clipwise_output']))
    if isnan == True:
        return statistics, output_dict
    # Clipwise statistics
    statistics['mse'] = metrics.mean_squared_error(
        output_dict['target'], output_dict['clipwise_output'], squared=False)
    (statistics['pearson_s'], statistics['p_value']) = pearsonr(
        output_dict['target'], output_dict['clipwise_output'])
    (statistics['SRCC'],_) = spearmanr(
        output_dict['target'], output_dict['clipwise_output'])

    return statistics, output_dict

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

def forward(model, data_loader, return_input=False, return_target=False):
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
    output_dict = {}
    total_time = 0
    # Evaluate on mini-batch
    for n, wav_dic in enumerate(data_loader):
        #print(batch_data_dict['audio_name'].shape)
        # Predict
        #print(wav_dic)
        (wav, target, feat, _) = wav_dic
        #print("label : " , target)
        batch_waveform = move_data_to_device(wav, device)
        batch_feat = move_data_to_device(feat, device)
        #batch_feat = feat
        

        with torch.no_grad():
            model.eval()
            s_t = time.time()
            batch_output, _ = model(batch_waveform, batch_feat)
            infer_time = time.time() - s_t

        total_time += infer_time

        print(infer_time)
        #batch_output = torch.clamp(batch_output, 1, 5)
        #print("predi : " ,batch_output)
        append_to_dict(output_dict, 'clipwise_output', 
            batch_output.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target)

    print(total_time / len(data_loader))    
        
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict

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

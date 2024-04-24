import numpy as np
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging
import torch
from torch.utils import data
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

from utilities import int16_to_float32
#import config
import json
import soundfile as sf
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model import load_best_model
from asteroid.filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag
from pypesq import pesq


class DSEDataset(data.Dataset):                                                                                          
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, conf):
    
        super(DSEDataset, self).__init__()                                                                               
        self.json_dir = json_dir                                                                                        
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)                                                                               
        
        self.wav_ids = list(self.mix_infos.keys())                                                                      
        self.model_en = load_best_model(conf, conf["main_args"]["exp_dir_se"])
        #if conf["main_args"]["use_gpu"]:
        #    self.model_en = self.model_en.cuda()
        self.model_en = self.model_en.cuda()
        self.model_device = next(self.model_en.parameters()).device
        self.stft, _ = make_enc_dec("stft", **conf["filterbank"])


        
    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):                                                                                         
        """Gets a mixture/sources pair.                                                                                 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        
        with torch.no_grad():
            #net_inp = x[None].to(self.model_device)
            net_inp = x[None].to(self.model_device)
            out, _ = self.model_en(net_inp)
            out = out.squeeze().cpu()
            #out = model(net_inp)
            #magnitude=mag(output)
        spec = self.stft(x[None])
        spec = mag(spec).squeeze().cpu()

        diff = spec - out
        
        # Load clean                                                
        mos = torch.tensor(utt_info["mos"]) 
        #print(self.wav_ids[idx])
        #print("diff", diff.size())
        return diff, mos
        # Load clean                                                                                                    
        #speech = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        # Load noise
        #noise = torch.from_numpy(sf.read(utt_info["noise"], dtype="float32")[0])                                       
    
        #return x, speech                                                                                                
class P563Dataset(data.Dataset):                                                                                          
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(P563Dataset, self).__init__()                                                                               
        self.json_dir = json_dir                                                                                        
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)                                                                               
        self.wav_ids = list(self.mix_infos.keys())                                                                      
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):                                                                                         
        """Gets a mixture/sources pair.                                                                                 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]        
        utt_info["mix"] = utt_info["mix"].replace("wav", "raw")
        # Load mixture
        main = 'p563/p563' + ' ' + str(utt_info["mix"])
        f = os.popen(main)
        i = 0
        for line in f.readlines():    
            print(line) 
            i=i+1
            if i==2:
                line = line.strip().split('\t')
                print(line[1])
        f.close()  
        data = torch.tensor(float(line[1]))
        
        mos = torch.tensor(utt_info["mos"]) 
        #print(utt_info["mix"])
        #print("label :", mos)
        if self.return_clean == False :
            return x, mos
        else :
            return data, mos

class TE2EDataset(data.Dataset):                                                                                          
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(TE2EDataset, self).__init__()                                                                               
        self.json_dir = json_dir                                                                                        
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)                                                                               
        self.wav_ids = list(self.mix_infos.keys())                                                                      
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):                                                                                         
        """Gets a mixture/sources pair.                                                                                 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        # Load noise
        #noise = torch.from_numpy(sf.read(utt_info["noise"], dtype="float32")[0])                                       
        mos = torch.tensor(utt_info["mos"]) 
        #print(utt_info["mix"])
        #print("label :", mos)
        if self.return_clean == False :
            return x, mos
        else :
            return x, clean, mos, utt_info["mix"]

class E2EDataset(data.Dataset):
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
        super(E2EDataset, self).__init__()
        self.json_dir = json_dir      
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)                                                                               
        self.wav_ids = list(self.mix_infos.keys())                                                                      
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):
        """Gets a mixture/sources pair.    
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])
        #print(x.shape)
        len_x = x.shape[0] / 16000
        #print(len_x)

        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        # Load noise
        #noise = torch.from_numpy(sf.read(utt_info["noise"], dtype="float32")[0])                                       
        mos = torch.tensor(utt_info["mos"]) 
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]
        if self.return_clean == False :
            #return x, mos, name
            return x, mos, len_x
        else :
            return x, clean, mos

class AUENDataset(data.Dataset): 
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(AUENDataset, self).__init__()  
        self.json_dir = json_dir    
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)   
        self.wav_ids = list(self.mix_infos.keys())
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):     
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        mos = torch.tensor(utt_info["mos"]) 
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]

        return x, mos, name

class DAAEDataset(data.Dataset): 
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(DAAEDataset, self).__init__()  
        self.json_dir = json_dir    
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)   
        self.wav_ids = list(self.mix_infos.keys())
        self.return_clean = return_clean
        
        self.speaker_list={}
        fin = open("dataset_own/mi_smi_polqa/mos_list","r")
        line = fin.readline()
        i = 0 
        while line: 
            items = line.strip().split('_')
            #print(items[1])
            key = items[0]
            if key not in self.speaker_list.keys():
                self.speaker_list[items[0]] = i
                i = i + 1
            #name[items[0]]=items[1]
            line = fin.readline() 
        print("speaker_number", len(self.speaker_list))

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):     
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        mos = torch.tensor(utt_info["mos"]) 
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]
        speakers = self.wav_ids[idx].strip().split('_')
        speaker = speakers[0]
        speaker_id = self.speaker_list[speaker]

        return x, mos, name, speaker_id

class E2EDataset_3t(data.Dataset): 
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(E2EDataset_3t, self).__init__()  
        self.json_dir = json_dir    
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)   
        self.wav_ids = list(self.mix_infos.keys())
        self.return_clean = return_clean
        
        self.speaker_list={}
        self.class_list={}
        fin = open("dataset_own/mi_smi_polqa/mos_list","r")
        line = fin.readline()
        i = 0 
        j = 0
        while line: 
            key = line.strip().split('.')[0]
            #print(items[1])
            it = key.strip().split('_')
            spk = it[0]
            cla = it[2]
            if spk not in self.speaker_list.keys():
                self.speaker_list[spk] = i
                i = i + 1
            if cla not in self.class_list.keys():
                self.class_list[cla] = j
                j = j + 1

            #name[items[0]]=items[1]
            line = fin.readline() 
        print("speaker_number", len(self.speaker_list))
        print("class_number", len(self.class_list))

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):     
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        mos = utt_info["mos"]
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]
        
        speaker = self.wav_ids[idx].strip().split('_')[0]
        speaker_id = self.speaker_list[speaker]
        
        cla = self.wav_ids[idx].strip().split('_')[2]
        cla_id = self.class_list[cla]

        return x, mos, name, speaker_id, cla_id

class DAAEDataset_3t(data.Dataset): 
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(DAAEDataset_3t, self).__init__()  
        self.json_dir = json_dir    
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)   
        self.wav_ids = list(self.mix_infos.keys())
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):     
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        mos = utt_info["mos"]
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]
        speaker_id = utt_info["spk"]
        cla = utt_info["cla"]

        return x, mos, name, speaker_id, cla

class DAAEDataset_test_3t(data.Dataset): 
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(DAAEDataset_test_3t, self).__init__()  
        self.json_dir = json_dir    
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)   
        self.wav_ids = list(self.mix_infos.keys())
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):     
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        mos = utt_info["mos"]
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]
        if ".wav" in name:
            name = name[:-4]
        speaker_id = 1
        cla = 1

        return x, mos, name, speaker_id, cla


class DAAEDataset_test(data.Dataset): 
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(DAAEDataset_test, self).__init__()  
        self.json_dir = json_dir    
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)   
        self.wav_ids = list(self.mix_infos.keys())
        self.return_clean = return_clean
    
    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):     
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        mos = torch.tensor(utt_info["mos"]) 
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]
        
        speaker_id = 0

        return x, mos, name, speaker_id

class FUDataset(data.Dataset):
    """Deep Noise Suppression (DNS) Challenge's dataset.
    Args    
        json_dir (str): path to the JSON directory (from the recipe).
    """
    def __init__(self, json_dir, feat_dir, return_clean = False):
        super(FUDataset, self).__init__()
        self.json_dir = json_dir
        self.feat_dir = feat_dir
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:
            self.mix_infos = json.load(f)
        self.wav_ids = list(self.mix_infos.keys())
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    def __getitem__(self, idx):
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]
        # Load mixture              
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])
        (filepath, tempfilename) = os.path.split(utt_info["mix"])
        (filename, extension) = os.path.splitext(tempfilename)

        filename = os.path.join(self.feat_dir, 'feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_daae_mosnetcnn_w0303_ft/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_ae_ft/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_daae_w0307_ft/feat', filename + ".npy")
        
        #filename = os.path.join('exp/train_wb_new_daae_3t_deep_w0608_a01_minscore_ft/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_daae_2t_sccla_deep_w0608_minscore_ft/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_daae_2t_scspk_deep_w06_a01_minscore_ft/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_daae_2t_spkcla_deep_w0608_a01_mincla_ft/feat', filename + ".npy")
        
        #filename = os.path.join('exp/train_wb_new_daae_3t_deep_w0608_a01_minscore_pesq_ft/feat', filename + ".npy")

        #filename = os.path.join('exp/train_wb_new_daae_ft_lossfix/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_daae_ft/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_vae_new_ft/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_vqvae_ft/feat', filename + ".npy")
        #filename = os.path.join('exp/train_wb_new_vae_ft/feat', filename + ".npy")
        
        #filename = os.path.join('exp/train_mi_smi_mi_smi_daae/feat', filename + ".npy")
        #filename = os.path.join('exp/train_mi_smi_mi_smi_auen/feat', filename + ".npy")
        #filename = os.path.join('exp/train_mi_smi_mi_smi_vae/feat', filename + ".npy")
        #filename = os.path.join('exp/train_mi_smi_mi_smi_vae_new/feat', filename + ".npy")
        #filename = os.path.join('exp/train_mi_smi_mi_smi_vqvae/feat', filename + ".npy")
        #filename = os.path.join('exp/train_mi_smi_mi_smi_vqvae_128/feat', filename + ".npy")

        #feat = torch.load(filename).cpu()
        feat = np.load(filename)
        feat = torch.from_numpy(feat)
        feat = feat.squeeze(0).transpose(0,1)
        #ivec = torch.from_numpy(standardization(ivec))
        #print(ivec.shape)

        mos = torch.tensor(utt_info["mos"])
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]
        return x, mos, feat, name

class VCCDataset(data.Dataset):                                                                                          
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(VCCDataset, self).__init__()                                                                               
        self.json_dir = json_dir                                                                                        
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)                                                                               
        self.wav_ids = list(self.mix_infos.keys())                                                                      
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):     
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        mos = torch.tensor(utt_info["mos"]) 
        #print(utt_info["mix"])
        #print("label :", mos)
        name = self.wav_ids[idx]
        return x, mos, name


class VCCDataset_mbnet(data.Dataset):                                                                                          
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_judge = True):
    
        super(VCCDataset_mbnet, self).__init__()                                                                               
        self.json_dir = json_dir                                                                                        
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)                                                                               
        self.wav_ids = list(self.mix_infos.keys())                                                                      
        self.return_judge = return_judge

		

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):     
        """Gets a mixture/sources pair. 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                           
        #clean = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        mos = torch.tensor(utt_info["mos"])

        if self.return_judge == True:
            judgescore = torch.tensor(utt_info["judgescore"]) 
            #judgeid = torch.tensor(utt_info["judgeid"])
            judgeid = torch.tensor(utt_info["judgeid"])
            #print("label :", mos)
        else:
            judgescore = 1
            judgeid = 1
        #name = self.wav_ids[idx]
        items = self.wav_ids[idx].split('.')
        name = items[0]
        return x, mos, judgescore, judgeid, name


class PESQE2EDataset(data.Dataset):                                                                                          
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir, return_clean = False):
    
        super(PESQE2EDataset, self).__init__()                                                                               
        self.json_dir = json_dir                                                                                        
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)                                                                               
        self.wav_ids = list(self.mix_infos.keys())                                                                      
        self.return_clean = return_clean

    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):                                                                                         
        """Gets a mixture/sources pair.                                                                                 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        wav = sf.read(utt_info["mix"], dtype="float32")[0]
        # Load clean                                           
        clean = sf.read(utt_info["clean"], dtype="float32")[0]
        # Load noise
        #noise = torch.from_numpy(sf.read(utt_info["noise"], dtype="float32")[0])                                       
        mos = torch.tensor(utt_info["mos"]) 
        
        (length_wav,) = wav.shape
        (length_clean,) = clean.shape
        if length_wav > length_clean:
            wav = wav[:length_clean]
        elif length_wav < length_clean:
            clean = clean[:length_wav]
        
        sr = 8000
        mos_pesq = pesq(clean, wav, sr)
        #print(utt_info["mix"])
        #print("label :", mos)
        wav = torch.from_numpy(wav)
        clean = torch.from_numpy(clean)
        
        if self.return_clean == False :
            return wav, mos_pesq
        else :
            return wav, clean, mos_pesq

class SEDataset(data.Dataset):                                                                                          
    """Deep Noise Suppression (DNS) Challenge's dataset.                                                                
    Args
        json_dir (str): path to the JSON directory (from the recipe).                                                   
    """
    def __init__(self, json_dir):
    
        super(SEDataset, self).__init__()                                                                               
        self.json_dir = json_dir                                                                                        
        with open(os.path.join(json_dir, "merged_file.json"), "r") as f:                                                 
            self.mix_infos = json.load(f)                                                                               
        
        self.wav_ids = list(self.mix_infos.keys())                                                                      
        
    def __len__(self):
        return len(self.wav_ids)
    
    def __getitem__(self, idx):                                                                                         
        """Gets a mixture/sources pair.                                                                                 
        Returns:
            mixture, vstack([source_arrays])
        """
        utt_info = self.mix_infos[self.wav_ids[idx]]                                                                    
        # Load mixture
        x = torch.from_numpy(sf.read(utt_info["mix"], dtype="float32")[0])                                              
        # Load clean                                                                                                    
        speech = torch.from_numpy(sf.read(utt_info["clean"], dtype="float32")[0])                                       
        # Load noise
        #noise = torch.from_numpy(sf.read(utt_info["noise"], dtype="float32")[0])                                       
    
        return x, speech                                                                                                

class SQTask4Dataset(object):
    def __init__(self):
        """DCASE 2017 Task 4 dataset."""
        pass
 
    def __getitem__(self, meta):
        """Get input and target data of an audio clip.

        Args:
          meta: dict, e.g., {'hdf5_path':, xxx.h5, 'index_in_hdf5': 34}

        Returns:
          data_dict: {'audio_name': str, 
                      'waveform': (audio_samples,), 
                      'target': (classes_num,), 
                      (ifexist) 'strong_target': (frames_num, classes_num)}
        """

        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            waveform = int16_to_float32(hf['waveform'][index_in_hdf5])
            target = hf['target'][index_in_hdf5].astype(np.float32)

            data_dict = {
                'audio_name': audio_name, 'waveform': waveform, 'target': target}

            #if 'strong_target' in hf.keys():
            #    strong_target = hf['strong_target'][index_in_hdf5].astype(np.float32)
            #    data_dict['strong_target'] = strong_target

        return data_dict

class SQ2Dataset(data.Dataset):
    def __init__(self, hdf5_path):
        super(SQ2Dataset, self).__init__()
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as hf:
            self.audios_num = len(hf['audio_mix_name'])

    def __len__(self):
        return self.audios_num

    def __getitem__(self, idx):
        """Get input and target data of an audio clip.
        Args:

        Returns:
        """
        index_in_hdf5 = idx

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_mix_name = hf['audio_mix_name'][index_in_hdf5].decode()
            feature_diff = torch.from_numpy(int16_to_float32(hf['feature_diff'][index_in_hdf5]))
            target = hf['target'][index_in_hdf5].astype(np.float32)

        return feature_diff, target

class SQ1Dataset(data.Dataset):
    def __init__(self, hdf5_path):
        super(SQ1Dataset, self).__init__()
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as hf:
            self.audios_num = len(hf['audio_mix_name'])

    def __len__(self):
        return self.audios_num

    def __getitem__(self, idx):
        """Get input and target data of an audio clip.
        Args:

        Returns:
        """
        index_in_hdf5 = idx

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_mix_name = hf['audio_mix_name'][index_in_hdf5].decode()
            waveform_mix = torch.from_numpy(int16_to_float32(hf['waveform_mix'][index_in_hdf5]))
            audio_clean_name = hf['audio_clean_name'][index_in_hdf5].decode()
            waveform_clean = torch.from_numpy(int16_to_float32(hf['waveform_clean'][index_in_hdf5]))
            target = hf['target'][index_in_hdf5].astype(np.float32)
        
        #print(audio_mix_name, audio_clean_name)

        return waveform_mix, waveform_clean, target


class TrainSampler(object):
    def __init__(self, hdf5_path, batch_size, random_seed=1234):
        """Training data sampler.
        
        Args:
          hdf5_path, str
          batch_size: int
          random_seed: int
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        with h5py.File(self.hdf5_path, 'r') as hf:
            self.audios_num = len(hf['audio_name'])

        logging.info('Training audio num: {}'.format(self.audios_num))
        self.audio_indexes = np.arange(self.audios_num)        
        self.random_state.shuffle(self.audio_indexes)

        self.pointer = 0

    def __iter__(self):
        """Generate batch meta.
        
        Returns: 
          batch_meta: [{'hdf5_path':, xxx.h5, 'index_in_hdf5': 34},
                       {'hdf5_path':, xxx.h5, 'index_in_hdf5': 12},
                       ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.audio_indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.audio_indexes)
                
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': self.audio_indexes[index]})
                i += 1

            yield batch_meta


class TestSampler(object):
    def __init__(self, hdf5_path, batch_size):
        """Testing data sampler.
        
        Args:
          hdf5_path, str
          batch_size: int
        """
        super(TestSampler, self).__init__()
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

        with h5py.File(self.hdf5_path, 'r') as hf:
            self.audios_num = len(hf['audio_name'])
            
        logging.info('Test audio num: {}'.format(self.audios_num))
        self.audio_indexes = np.arange(self.audios_num)        

    def __iter__(self):
        """Generate batch meta for test. 
        
        Returns: 
          batch_meta: [{'hdf5_path':, xxx.h5, 'index_in_hdf5': 34},
                       {'hdf5_path':, xxx.h5, 'index_in_hdf5': 12},
                       ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': self.audio_indexes[index]})

            pointer += batch_size
            yield batch_meta


def collate_fn(list_data_dict):
    """Collate data.

    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (audio_samples,), ...}, 
                             {'audio_name': str, 'waveform': (audio_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, audio_samples), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict

import argparse
import json
import os
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from utils.losses import get_loss_func

from asteroid.utils import str2bool_arg
#from model import e2e_make_model_and_optimizer, auen_SimpleSystem
from model import e2e_make_model_and_optimizer, daae_3t_SimpleSystem
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils')) 

from data_generator import DAAEDataset_test_3t
import random
from asteroid import torch_utils
import numpy as np


random_seed = 17     
#os.environ['PYTHONHASHSEED'] = str(random_seed)
random.seed(random_seed)   
np.random.seed(random_seed)
torch.manual_seed(random_seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(random_seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(random_seed)

#torch.backends.cudnn.benchmark = False    
#torch.backends.cudnn.deterministic = True

#torch.manual_seed(17)  # Reproducibility on the dataset spliting

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
parser.add_argument("--pre_dir", default="exp/tmp")
parser.add_argument("--is_complex", default=False, type=str2bool_arg)
parser.add_argument("--model_type", default=None, type=str)
#parser.add_argument("--json_dir", default="./data", help="Where to save the json file")

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
    def _init_fn(seed):
        np.random.seed(17)
    #total_set = DNSDataset(conf["data"]["json_dir"])
    print(conf["data"]["json_dir"])
    train_set = DAAEDataset_test_3t(os.path.join(conf["data"]["json_dir"], "train/"))
    print(len(train_set))
    val_set = DAAEDataset_test_3t(os.path.join(conf["data"]["json_dir"], "val/"))

    train_loader = DataLoader(
        train_set,
        #shuffle=True,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=False,
        worker_init_fn=_init_fn,
		collate_fn=collate_fn_pad
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=False,
        worker_init_fn=_init_fn,
		collate_fn=collate_fn_pad
    )

    # Define model and optimizer in a local function (defined in the recipe).
    # Two advantages to this : re-instantiating the model and optimizer
    # for retraining and evaluating is straight-forward.
    model, optimizer = e2e_make_model_and_optimizer(conf)
    # Last best model summary
    pre_dir = conf["main_args"]["pre_dir"]
    with open(os.path.join(pre_dir, "best_k_models.json"), "r") as f:
        best_k = json.load(f)
    best_model_path = min(best_k, key=best_k.get)
    #best_model_path = max(best_k, key=best_k.get)
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location="cpu")
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint["state_dict"], model)
    model.train()
    
    for name, p in model.masker.named_parameters():
        if "mos_pred" in name or "cla_cla" in name or "spk_cla" in name :
            p.requires_grad = False
        #if "spk_cla" in name:
        #    p.requires_grad = False
    #model.masker.mos_rnn.weight.requires_grad = False
    #model.masker.spk_rnn.weight.requires_grad = False


    #model, optimizer = e2e_make_model_and_optimizer(conf)
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5, verbose=True)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    #loss_func = get_loss_func('clip_mse')
    #loss_func = get_loss_func('vae_loss_function')
    #loss_func = get_loss_func('vqvae_loss_function')
    #loss_func = get_loss_func('mse_ce')
    loss_func = get_loss_func('mse_3t')
    #system = auen_SimpleSystem(
    system = daae_3t_SimpleSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor="val_loss", mode="min", save_top_k=3, verbose=True)
    #checkpoint = ModelCheckpoint(checkpoint_dir, monitor="val_pearsonr", mode="max", save_top_k=3, verbose=True)
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=True))
        #callbacks.append(EarlyStopping(monitor="val_pearsonr", mode="max", patience=20, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        #distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)
    #print('grad1', model.mos_pred.weight.requires_grad)
    #print('grad2', model.spk_cla.weight.requires_grad)


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    #with open("local/conf_e2e_wb_ft.yml") as f:
    with open("local/conf_ae_ft_qq.yml") as f:
    #with open("local/conf_e2e_qq_len_ft.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)

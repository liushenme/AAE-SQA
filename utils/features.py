import os
import sys
import numpy as np
import argparse
import h5py
import librosa
import matplotlib.pyplot as plt
import time
import csv
import math
import re
import random

import config
from utilities import create_folder, pad_truncate_sequence, float32_to_int16


def get_weak_csv_filename(data_type):
    """Prepare weakly labelled csv path. 

    Args:
      data_type: 'training' | 'testing' | 'evaluation'

    Returns:
      str, weakly labelled csv path
    """
    if data_type in ['training', 'testing', 'evaluation']:
        return '{}_de_clean_score.csv'.format(data_type)
        
    else:
        raise Exception('Incorrect argument!')


def read_weak_csv(weak_label_csv_path, data_type):
    """Read weakly labelled ground truth csv file. There can be multiple labels
    for each audio clip.

    Args:
      weak_label_csv_path: str
      data_type: 'training' | 'testing' | 'evaluation'

    Returns:
      meta_list: [{'audio_name': 'a.wav', 'labels': ['Train', 'Bus']},
                  ...]
    """
    assert data_type in ['training', 'testing', 'evaluation']
    
    if data_type in ['training', 'testing', 'evaluation']:
        with open(weak_label_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            rows = list(reader)
            
    meta_list = []
            
    for row in rows:
        if data_type in ['training', 'testing', 'evaluation']:
            """row: ['3aEAf1s1.c08','4.625']"""
            meta = {
                'audio_mix_name':  row[0] + '.wav', 
                'audio_clean_name':  row[1] + '.wav',
                'labels':  row[2] }
            meta_list.append(meta)
        
    return meta_list


def read_strong_csv(strong_meta_csv_path):
    """Read strongly labelled ground truth csv file. 
    
    Args:
      strong_meta_csv_path: str

    Returns: 
      meta_dict: {'a.wav': [{'onset': 3.0, 'offset': 5.0, 'label': 'Bus'},
                            {'onset': 4.0, 'offset': 7.0, 'label': 'Train'}
                            ...],
                  ...}
    """
    with open(strong_meta_csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter='\t')
        lines = list(reader)
        
    meta_dict = {}
    for line in lines:
        """line: ['-5QrBL6MzLg_60.000_70.000.wav', '0.917', '2.029', 'Train horn']"""
        [audio_name, onset, offset, label] = line
        meta = {'onset': onset, 'offset': offset, 'label': label}
        if audio_name in meta_dict:
            meta_dict[audio_name].append(meta)
        else:
            meta_dict[audio_name] = [meta]
        
    return meta_dict


def get_weak_target(labels, lb_to_idx):
    """Labels to vector. 

    Args:
      labels: list of str
      lb_to_idx: dict

    Returns:
      target: (classes_num,)
    """
    classes_num = len(lb_to_idx)
    target = np.zeros(classes_num, dtype=np.bool)
    
    for label in labels: 
        target[lb_to_idx[label]] = 1.
        
    return target 


def get_strong_target(audio_name, strong_meta_dict, frames_num, 
    frames_per_second, lb_to_idx):
    """Reformat strongly labelled target to matrix format. 

    Args:
      audio_name: str
      strong_meta_dict: dict, e.g., 
          {'a.wav': [{'onset': 3.0, 'offset': 5.0, 'label': 'Bus'},
                     {'onset': 4.0, 'offset': 7.0, 'label': 'Train'}
                      ...],
           ...}
      frames_num: int
      frames_per_second: int
      lb_to_idx: dict

    Returns:
      target: (frames_num, classes_num)
    """
    
    meta_list = strong_meta_dict[audio_name]
    
    target = np.zeros((frames_num, len(lb_to_idx)), dtype=np.bool)
    
    for meta in meta_list:
        onset = float(meta['onset']) 
        bgn_frame = int(round(onset * frames_per_second))
        offset = float(meta['offset'])
        end_frame = int(round(offset * frames_per_second)) + 1
        label = meta['label']
        idx = lb_to_idx[label]
        
        target[bgn_frame : end_frame, idx] = 1
    
    return target


def pack_audio_files_to_hdf5(args):
    """Pack waveform to hdf5 file. 

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, Directory of your workspace
      data_type: 'training' | 'testing' | 'evaluation'
      mini_data: bool, set True for debugging on a small part of data
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    audio_samples = config.audio_samples
    #classes_num = config.classes_num
    #lb_to_idx = config.lb_to_idx
    #frames_per_second = config.frames_per_second
    #frames_num = frames_per_second * config.audio_duration + 1
    """The +1 frame comes from the 'center=True' argument when extracting spectrogram."""

    #has_strong_target = data_type in ['testing', 'evaluation']

    # Paths
    #audios_dir = os.path.join(dataset_dir, data_type)
    audios_mix_dir = os.path.join(dataset_dir, data_type)
    audios_clean_dir = os.path.join(dataset_dir, data_type+'_clean')
    weak_label_csv_path = os.path.join(dataset_dir, 'metadata', 
        get_weak_csv_filename(data_type))

    #if data_type == 'testing':
    #    strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 
    #        'groundtruth_strong_label_testing_set.csv')
    #elif data_type == 'evaluation':
    #    strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 
    #        'groundtruth_strong_label_evaluation_set.csv')

    if mini_data:
        packed_hdf5_path = os.path.join(workspace, 'hdf5s', 
            'minidata_{}.h5'.format(data_type))
    else:
        packed_hdf5_path = os.path.join(workspace, 'hdf5s', 
            '{}.h5'.format(data_type))
    create_folder(os.path.dirname(packed_hdf5_path))

    # Read metadata
    weak_meta_list = read_weak_csv(weak_label_csv_path, data_type)
    """e.g., [{'audio_mix_name': 'a.wav','audio_clean_name': 'a.wav', 'labels': '4.625'},
              ...]"""

    # Use a small amount of data for debugging
    if mini_data:
        random.seed(1234)
        random.shuffle(weak_meta_list)
        weak_meta_list = weak_meta_list[0 : 100]

    audios_num = len(weak_meta_list)

    audio_indexes = np.arange(audios_num)
    
    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_mix_name', 
            shape=(audios_num,), 
            dtype='S80')

        hf.create_dataset(
            name='waveform_mix', 
            shape=(audios_num, audio_samples), 
            dtype=np.int16)
        
        hf.create_dataset(
            name='audio_clean_name', 
            shape=(audios_num,), 
            dtype='S80')

        hf.create_dataset(
            name='waveform_clean', 
            shape=(audios_num, audio_samples), 
            dtype=np.int16)

        hf.create_dataset(
            name='target', 
            shape=(audios_num,), 
            dtype=np.float32)
        
        hf.create_dataset(
            name='index', 
            shape=(audios_num,), 
            dtype=np.int16)

        for n in range(audios_num):
            print(n)
            weak_meta_dict = weak_meta_list[n]
            audio_mix_name = weak_meta_dict['audio_mix_name']
            audio_mix_path = os.path.join(audios_mix_dir, audio_mix_name)
            (audio_mix, fs) = librosa.core.load(audio_mix_path, sr=sample_rate, mono=True)
            #audio_mix = pad_truncate_sequence(audio_mix, audio_samples)
            print(audio_mix_name)
            print("audio_mix=",audio_mix.shape)
            audio_clean_name = weak_meta_dict['audio_clean_name']
            audio_clean_path = os.path.join(audios_clean_dir, audio_clean_name)
            (audio_clean, fs) = librosa.core.load(audio_clean_path, sr=sample_rate, mono=True)
            #audio_clean = pad_truncate_sequence(audio_clean, audio_samples)
            print(audio_clean_name)
            print("audio_mix=",audio_clean.shape)
            assert audio_mix.shape == audio_clean.shape

            hf['audio_mix_name'][n] = audio_mix_name.encode()
            hf['waveform_mix'][n] = float32_to_int16(audio_mix)
            hf['audio_clean_name'][n] = audio_clean_name.encode()
            hf['waveform_clean'][n] = float32_to_int16(audio_clean)
            hf['target'][n] = np.float32(weak_meta_dict['labels'])
            #hf['index'][n] = np.int16(audio_indexes[n])
            #print(type(np.float32(weak_meta_dict['labels'])))
            #if has_strong_target:
            #    strong_target = get_strong_target(
            #        weak_meta_dict['audio_name'][1:], strong_meta_dict, 
            #        frames_num, frames_per_second, lb_to_idx)
                
            #    hf['strong_target'].resize((n + 1, frames_num, classes_num))
            #    hf['strong_target'][n] = strong_target

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Pack waveform to hdf5 file
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--data_type', type=str, choices=['training', 'testing', 'evaluation'], required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
        
    else:
        raise Exception('Incorrect arguments!')

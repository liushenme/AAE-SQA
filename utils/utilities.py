import os
import sys
import librosa
import logging
import matplotlib.pyplot as plt
import datetime
import pickle
import numpy as np
import csv

#import utils.config
#import config
#from utils.vad import activity_detection
#from vad import activity_detection


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def float32_to_int16(x):
    if np.max(np.abs(x)) > 1.:
        x /= np.max(np.abs(x))
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)



def write_submission(event_list, submission_path):
    """Write prediction event list to submission file for later evaluation.

    Args:
      event_list: list of events
      submission_path: str
    """
    f = open(submission_path, 'w')
            
    for event in event_list:
        f.write('{}\t{}\t{}\t{}\n'.format(
            event['filename'][1:], event['onset'], event['offset'], event['event_label']))
            
    logging.info('    Write submission file to {}'.format(submission_path))


def official_evaluate(reference_csv_path, prediction_csv_path):
    """Evaluate metrics with official SED toolbox. 

    Args:
      reference_csv_path: str
      prediction_csv_path: str
    """
    reference_event_list = sed_eval.io.load_event_list(reference_csv_path, 
        delimiter='\t', csv_header=False, 
        fields=['filename','onset','offset','event_label'])

    estimated_event_list = sed_eval.io.load_event_list(prediction_csv_path, 
        delimiter='\t', csv_header=False, 
        fields=['filename','onset','offset','event_label'])
    
    evaluated_event_labels = reference_event_list.unique_event_labels
    files={}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))
    
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=evaluated_event_labels,
        time_resolution=1.0
    )

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        segment_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )
    results = segment_based_metrics.results()

    return results


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        #self.statistics_dict = {'train': [], 'test': [], 'evaluate': []}
        self.statistics_dict = {'train': [], 'test': []}

    def append(self, data_type, iteration, statistics):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))

    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

        #resume_statistics_dict = {'train': [], 'test': [], 'evaluate': []}
        resume_statistics_dict = {'train': [], 'test': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict
 

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.

        Args:
          batch_size: int

        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)

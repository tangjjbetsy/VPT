import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import librosa
import logging
import wandb
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, RegressionPostProcessor) 
from data_generator import MaestroDataset, HackkeyDataset, HackkeyMidiDataset, Augmentor, Sampler, TestSampler, collate_fn
from models import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN, Regress_onset_offset_frame_velocity_hackkey_CRNN
from pytorch_utils import move_data_to_device
from losses import get_loss_func
from evaluate import SegmentEvaluator
from tqdm import tqdm
import config

def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
      loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """

    # Arugments & parameters
    datadir = args.datadir
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data
    filename = args.filename

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8

    # Loss function
    loss_func = get_loss_func(loss_type)

    # Paths
    # hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')
    hdf5s_dir = os.path.join(datadir, 'hdf5s', 'hackkey')
    # hdf5s_dir = os.path.join(datadir, 'hdf5s', 'hackkey_midi')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift),
        'batch_size={}'.format(batch_size),
        'time={}'.format(time.strftime('%Y-%m-%d_%H-%M-%S')))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size))
    create_folder(logs_dir)

    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(frames_per_second=frames_per_second, classes_num=classes_num)
    
    # Load full checkpoint
    checkpoint = torch.load("checkpoints/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth", map_location=device, weights_only=False)
    
    # Filter out velocity_model layers
    # filtered_state_dict = {k: v for k, v in checkpoint['model']['note_model'].items() if not k.startswith("velocity_model.")}
        
    # Load the filtered state dict insto the model
    # model.load_state_dict(filtered_state_dict, strict=False)
    model.load_state_dict(checkpoint['model']['note_model'], strict=False)
    logging.info('Model loaded from {}'.format("checkpoints/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"))

    # Freeze some layers
    for name, param in model.named_parameters():
        if 'hackkey_model' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            logging.info(f"Training parameter: {name}")
    
    # Wandb Logger
    if mini_data == False:
        # experiment = wandb.init(project='Hackkey_Kong', resume='allow', anonymous='must')
        # experiment.config.update(
        #     dict(model_type=model_type, loss_type=loss_type, augmentation=augmentation,
        #         max_note_shift=max_note_shift, mini_data=mini_data, 
        #         batch_size=batch_size, learning_rate=learning_rate,
        #         reduce_iteration=reduce_iteration, resume_iteration=resume_iteration,
        #         early_stop=early_stop, segment_seconds=segment_seconds,
        #         hop_seconds=hop_seconds, frames_per_second=frames_per_second,
        #         classes_num=classes_num, sample_rate=sample_rate,
        #         segment_samples=segment_samples, num_workers=num_workers,
        #         hdf5s_dir=hdf5s_dir, checkpoints_dir=checkpoints_dir,
        #         statistics_path=statistics_path, logs_dir=logs_dir,
        #         device=device, filename=filename)
        #     )
        pass

    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect argumentation!')

    train_dataset = HackkeyDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=max_note_shift, augmentor=augmentor)    

    # Sampler for training
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train', 
        segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    for batch_data_dict in tqdm(train_loader):
        pass
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True, choices=['none', 'aug'])
    parser_train.add_argument('--max_note_shift', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--datadir', type=str, required=True)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')